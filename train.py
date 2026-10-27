"""
Assignment 2: training pipeline.

Forest-fire surveillance cameras around Madrid: one image per camera per minute,
and the model decides whether to alert Madrid 112. Run via
`uv run python main.py train`. Stages:

1. Download the dataset version from Roboflow Universe (as is, not forked).
2. Describe the dataset: instances per class, split sizes, object size.
3. Finetune `yolo11s` from the pretrained COCO checkpoint.
4. Evaluate on the test split: per-class detection metrics, confusion matrix.
5. Turn detections into one image-level fire score per image, and compare
   against the COCO zero-shot and predict-nothing baselines.
6. Sweep the alert threshold against the cost matrix at the production base rate.
7. Search the two thresholds (pass / human review / alert) and size the review team.
8. Save everything `app.py` needs, so the dashboard never trains.

Thresholds are chosen on the validation split and only reported on test, so
the test numbers are not tuned to the test images.

Set `SMOKE_TEST=1` to run one short epoch on a slice of the data, writing to
`runs/smoke/` instead of `weights/` and `artifacts/`. Set `EVAL_ONLY=1` to skip
training and rerun stages 4-8 from the last `runs/finetune` checkpoint.

The pipeline runs under `if __name__ == "__main__"` because Windows starts
dataloader workers by re-importing this file; without the guard each worker
would re-run the whole pipeline.
"""

import json
import math
import os
import shutil
from pathlib import Path

import truststore

# Verify HTTPS against the OS certificate store, so downloads work behind
# antivirus/proxy TLS inspection as well.
truststore.inject_into_ssl()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

# Dataset: https://universe.roboflow.com/auebmsc-ba-machine-learning/forest-fire-and-smoke-detection-using-uav-imaging_3/dataset/9
WORKSPACE = "auebmsc-ba-machine-learning"
PROJECT = "forest-fire-and-smoke-detection-using-uav-imaging_3"
VERSION = 9
# Kept short on purpose: the repo path is already ~100 chars, and Windows caps
# full paths at 260.
DATA_DIR = ROOT / "datasets" / f"uav-v{VERSION}"
DATA_YAML = DATA_DIR / "data_local.yaml"

SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"
EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"
OUT_DIR = ROOT / "runs" / "smoke" if SMOKE_TEST else ROOT
WEIGHTS_DIR = OUT_DIR / "weights"
ARTIFACTS_DIR = OUT_DIR / "artifacts"
RUNS_DIR = ROOT / "runs"

BASE_MODEL = "yolo11s.pt"
EPOCHS = 1 if SMOKE_TEST else 100
PATIENCE = 20
IMGSZ = 640
BATCH = 8  # what fits in the 4 GB of the RTX 3050 Laptop GPU used for training
TRAIN_FRACTION = 0.02 if SMOKE_TEST else 1.0
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 6
SEED = 0

# ---------------------------------------------------------------------------
# Business assumptions. Every figure is argued in REPORT.md. Costs are in euros
# per case, where a case is one image from one camera.
# ---------------------------------------------------------------------------

C_FP = 1_000  # ground crew sent to a fire that is not there
C_TP = 1_000  # same mobilisation, but to a real fire
C_TN = 0
# FN = extra escape probability x hectares of an escaped fire x extinction EUR/ha.
C_FN = round(0.10 * 10 * 10_000)
# Sensitivity: the FN cost under other assumptions (FN/FP ratio in the label).
FN_SCENARIOS = {
    "0.6x (0.05 x 4 ha x 3k/ha)": 600,
    "5x": 5_000,
    "10x (central)": C_FN,
    "50x": 50_000,
    "190x": 190_000,
}

# Base rate: how many images actually show a fire.
CAMERAS = 34  # one camera per existing INFOMA watchtower
IMAGES_PER_CAMERA_PER_DAY = 24 * 60  # one image a minute, 24 h
FIRES_PER_YEAR = 265  # Comunidad de Madrid, 2024
CAMERAS_PER_FIRE = 2  # overlapping ~15 km tower ranges
MINUTES_VISIBLE = 30  # until the fire is confirmed or put out
CASES_PER_DAY = CAMERAS * IMAGES_PER_CAMERA_PER_DAY
FIRE_IMAGES_PER_YEAR = FIRES_PER_YEAR * CAMERAS_PER_FIRE * MINUTES_VISIBLE
BASE_RATE = FIRE_IMAGES_PER_YEAR / (CASES_PER_DAY * 365)

# Human review by a Madrid 112 operator looking at the camera feed.
REVIEW_SECONDS = 30
REVIEWER_COST_PER_HOUR = 25
REVIEWERS_AVAILABLE = 2  # operators on shift at any moment, 24 h
REVIEW_COST = REVIEW_SECONDS / 3600 * REVIEWER_COST_PER_HOUR

# 1.01 is above any confidence: "never alert".
THRESHOLDS = np.round(np.arange(0.0, 1.0101, 0.01), 2)


# ---------------------------------------------------------------------------
# 1. Download the dataset
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download the YOLO-format export once; reuse it on later runs."""
    done = DATA_DIR / ".complete"
    if done.is_file():
        print(f"Dataset already at {DATA_DIR}, skipping download.")
        return DATA_DIR
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov11", location=str(DATA_DIR), overwrite=True)
    done.touch()
    return DATA_DIR


def normalise_labels() -> int:
    """Rewrite polygon rows as their bounding box.

    63 label files mix polygon and box rows, and Ultralytics silently drops
    any image whose labels mix the two. Taking the polygon's bounding box is
    what Ultralytics itself does for polygon-only files, so no image is lost and
    no box changes. Idempotent: a second run finds only boxes.
    """
    rewritten = 0
    for label in DATA_DIR.glob("*/labels/*.txt"):
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if all(len(r) == 5 for r in rows):
            continue
        boxes = []
        for r in rows:
            if len(r) == 5:
                boxes.append(" ".join(r))
                continue
            xs, ys = list(map(float, r[1::2])), list(map(float, r[2::2]))
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            boxes.append(f"{r[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        label.write_text("\n".join(boxes) + "\n")
        rewritten += 1
    for cache in DATA_DIR.glob("*/labels.cache"):
        cache.unlink()
    return rewritten


def write_data_yaml() -> list[str]:
    """Point Ultralytics at absolute split paths; return the class names.

    Roboflow writes split paths relative to its own layout, which breaks when
    the working directory changes.
    """
    class_names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    DATA_YAML.write_text(yaml.safe_dump({
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(class_names)),
    }))
    return class_names


# ---------------------------------------------------------------------------
# 2. Describe the dataset
# ---------------------------------------------------------------------------

def read_labels(split: str):
    """Yield (image path, [(class, box width, box height), ...]) per image."""
    for image in sorted((DATA_DIR / split / "images").iterdir()):
        label = DATA_DIR / split / "labels" / f"{image.stem}.txt"
        boxes = []
        if label.is_file():
            for line in label.read_text().splitlines():
                values = line.split()
                if values:
                    boxes.append((int(values[0]), float(values[3]), float(values[4])))
        yield image, boxes


def describe_split(split: str, class_names: list[str]) -> list[dict]:
    images = list(read_labels(split))
    rows = [{
        "split": split, "class": "(all images)", "images": len(images),
        "background_images": sum(not boxes for _, boxes in images),
    }]
    for i, name in enumerate(class_names):
        areas = [w * h for _, boxes in images for c, w, h in boxes if c == i]
        rows.append({
            "split": split, "class": name,
            "instances": len(areas),
            "images_with_class": sum(any(c == i for c, _, _ in boxes) for _, boxes in images),
            "median_box_area": float(np.median(areas)) if areas else 0.0,
            "share_boxes_below_1pct": float(np.mean(np.array(areas) < 0.01)) if areas else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# 3-4. Finetune and evaluate detection
# ---------------------------------------------------------------------------

def finetune() -> Path:
    best = WEIGHTS_DIR / "best.pt"
    if EVAL_ONLY:
        shutil.copy(RUNS_DIR / "finetune" / "weights" / "best.pt", best)
        print(f"EVAL_ONLY: reusing runs/finetune checkpoint as {best}")
        return best
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        fraction=TRAIN_FRACTION,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        project=str(RUNS_DIR),
        name="smoke_train" if SMOKE_TEST else "finetune",
        exist_ok=True,
    )
    # The AMP check downloads a small pretrained model into ./weights, which
    # is committed and scanned by check.py; only our own weights belong there.
    for checkpoint in (ROOT / "weights").glob("yolo*.pt"):
        checkpoint.unlink()
    shutil.copy(model.trainer.best, best)
    print(f"Saved {best}")
    return best


def evaluate_detection(weights: Path, class_names: list[str]):
    """Per-class AP and the detection confusion matrix on the test split."""
    metrics = YOLO(str(weights)).val(
        data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        workers=WORKERS, project=str(RUNS_DIR),
        name="smoke_test" if SMOKE_TEST else "test_finetuned", exist_ok=True,
    )
    per_class = pd.DataFrame([
        {
            "class": class_names[c],
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "ap50": float(metrics.box.ap50[i]),
            "ap50_95": float(metrics.box.ap[i]),
        }
        for i, c in enumerate(metrics.box.ap_class_index)
    ])

    # The confusion matrix uses the validation confidence (0.001 by default,
    # right for mAP but meaningless as a matrix), so build it again at 0.25.
    # plots must stay on: Ultralytics only fills the matrix when plotting.
    cm_metrics = YOLO(str(weights)).val(
        data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        workers=WORKERS, conf=0.25, verbose=False,
        project=str(RUNS_DIR), name="test_confusion", exist_ok=True,
    )
    # Ultralytics stores the matrix as [predicted, true].
    confusion = {
        "labels": [*class_names, "background"], "rows": "predicted", "columns": "true",
        "matrix": cm_metrics.confusion_matrix.matrix.astype(int).tolist(),
    }
    return metrics, per_class, confusion


# ---------------------------------------------------------------------------
# 5. Image-level scores and baselines
# ---------------------------------------------------------------------------

def image_scores(weights: Path, class_names: list[str], split: str) -> pd.DataFrame:
    """One row per image of `split`: whether it shows a fire, and the model's score.

    The positive class is "the image shows a fire" (any fire or smoke box). The
    score is the highest confidence among the image's fire and smoke detections.
    """
    detector = YOLO(str(weights))
    rows = []
    for image, boxes in read_labels(split):
        result = detector.predict(str(image), imgsz=IMGSZ, conf=0.001, device=DEVICE, verbose=False)[0]
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        true_classes = {c for c, _, _ in boxes}
        row = {"image": image.name, "y_true": int(bool(boxes))}
        for i, name in enumerate(class_names):
            row[f"true_{name}"] = int(i in true_classes)
            row[f"score_{name}"] = float(confs[classes == i].max()) if (classes == i).any() else 0.0
        row["score"] = max(row[f"score_{name}"] for name in class_names)
        rows.append(row)
    return pd.DataFrame(rows)


def coco_zero_shot_findings() -> pd.Series:
    """What the COCO model sees in fire images.

    It has no fire or smoke class, so it can never raise an alert: its fire score
    is 0 for every image. This records what it detects instead.
    """
    coco = YOLO(BASE_MODEL)
    return pd.Series([
        coco.names[int(c)]
        for image, boxes in read_labels("test") if boxes
        for c in coco.predict(str(image), imgsz=IMGSZ, conf=0.25, device=DEVICE, verbose=False)[0].boxes.cls
    ], dtype=str).value_counts()


def summarise(name: str, score: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    alert = score >= threshold
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    fn, tn = int((~alert & (y == 1)).sum()), int((~alert & (y == 0)).sum())
    return {
        "model": name, "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn),
        "false_alarm_rate": fp / (fp + tn),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "accuracy": (tp + tn) / len(score),
    }


# ---------------------------------------------------------------------------
# 6. Cost against threshold
# ---------------------------------------------------------------------------

def expected_cost(tpr, fpr, c_fn=C_FN, base_rate=BASE_RATE):
    """Expected cost per case, reweighting test-split rates to the production base rate.

    The test split is about half fire images; production is about 1 in 1,100.
    """
    return base_rate * (tpr * C_TP + (1 - tpr) * c_fn) + (1 - base_rate) * (fpr * C_FP + (1 - fpr) * C_TN)


def cost_sweep(s: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    sweep = pd.DataFrame({"threshold": THRESHOLDS})
    sweep["tpr"] = [(s[y == 1] >= t).mean() for t in THRESHOLDS]
    sweep["fpr"] = [(s[y == 0] >= t).mean() for t in THRESHOLDS]
    sweep["alerts_per_day"] = (BASE_RATE * sweep["tpr"] + (1 - BASE_RATE) * sweep["fpr"]) * CASES_PER_DAY
    for label, c_fn in FN_SCENARIOS.items():
        sweep[f"cost_per_1000 {label}"] = 1000 * expected_cost(sweep["tpr"], sweep["fpr"], c_fn)
    return sweep


def plot_cost_curve(sweep: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in FN_SCENARIOS:
        ax.plot(sweep["threshold"], sweep[f"cost_per_1000 {label}"], label=f"FN cost {label}",
                lw=2.5 if "central" in label else 1)
    ax.axvline(chosen, color="k", ls="--", label=f"chosen t={chosen:.2f}")
    ax.set(xlabel="alert threshold", ylabel="expected cost per 1,000 images (EUR)", yscale="log",
           title="Cost against threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# 7. Pass, human review, or alert
# ---------------------------------------------------------------------------

def band_rates(score: np.ndarray, t_low: float, t_high: float) -> tuple[float, float, float]:
    """Share of cases that pass, go to review, and alert."""
    alert = (score >= t_high).mean()
    passed = (score < t_low).mean()
    return passed, 1 - alert - passed, alert


def upper_rate(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate seen k times in n.

    With k = 0 this is about 3 / n (the rule of three): seeing no false alarm
    on a few hundred negatives does not show the rate is zero.
    """
    def at_most_k(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at_most_k(mid) > 1 - confidence else (lo, mid)
    return hi


def policy(s: np.ndarray, y: np.ndarray, t_low: float, t_high: float, c_fn: float = C_FN) -> dict:
    """Cost and review load of: pass below t_low, alert at or above t_high, review between.

    The operator is assumed to get it right, so a reviewed fire costs
    C_TP + review and a reviewed non-fire costs only the review.
    """
    p_pass, p_rev, p_alert = band_rates(s[y == 1], t_low, t_high)
    n_pass, n_rev, n_alert = band_rates(s[y == 0], t_low, t_high)
    cost = (
        BASE_RATE * (p_alert * C_TP + p_rev * (C_TP + REVIEW_COST) + p_pass * c_fn)
        + (1 - BASE_RATE) * (n_alert * C_FP + n_rev * (C_TN + REVIEW_COST) + n_pass * C_TN)
    )
    review_share = BASE_RATE * p_rev + (1 - BASE_RATE) * n_rev
    reviews_per_day = review_share * CASES_PER_DAY
    reviewer_hours = reviews_per_day * REVIEW_SECONDS / 3600
    negatives = int((y == 0).sum())
    false_alerts_upper = upper_rate(int(round(n_alert * negatives)), negatives) if t_high <= 1 else 0.0
    return {
        "t_low": t_low, "t_high": t_high,
        "cost_per_1000": 1000 * cost, "cost_per_day": cost * CASES_PER_DAY,
        "review_share": review_share, "reviews_per_day": reviews_per_day,
        "reviewers_on_shift": reviewer_hours / 24,
        "fires_missed_share": p_pass,
        "false_alerts_per_day": (1 - BASE_RATE) * n_alert * CASES_PER_DAY,
        "false_alerts_per_day_upper95": (1 - BASE_RATE) * false_alerts_upper * CASES_PER_DAY,
    }


if __name__ == "__main__":
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} smoke_test={SMOKE_TEST} epochs={EPOCHS}")
    print(f"cases/day={CASES_PER_DAY:,}  base rate={BASE_RATE:.5f} (1 in {1 / BASE_RATE:,.0f})")
    print(f"review cost per case={REVIEW_COST:.2f} EUR")

    # 1. Data
    download_dataset()
    print(f"Converted polygons to boxes in {normalise_labels()} label files")
    class_names = write_data_yaml()

    # 2. Describe
    dataset_stats = pd.DataFrame([
        row for split in ("train", "valid", "test") for row in describe_split(split, class_names)
    ])
    dataset_stats.to_csv(ARTIFACTS_DIR / "dataset_stats.csv", index=False)
    print(dataset_stats.to_string(index=False))

    # 3-4. Finetune, then detection metrics on test
    best_weights = finetune()
    test_metrics, per_class, confusion = evaluate_detection(best_weights, class_names)
    per_class.to_csv(ARTIFACTS_DIR / "per_class.csv", index=False)
    (ARTIFACTS_DIR / "detection_confusion.json").write_text(json.dumps(confusion, indent=2))
    print(per_class.to_string(index=False))
    print(pd.DataFrame(
        confusion["matrix"],
        index=[f"pred {l}" for l in confusion["labels"]],
        columns=[f"true {l}" for l in confusion["labels"]],
    ))

    # 5. Image-level scores and baselines
    # Validation scores pick the thresholds; test scores only report them.
    val_scores = image_scores(best_weights, class_names, "valid")
    val_scores.to_csv(ARTIFACTS_DIR / "valid_scores.csv", index=False)
    y_val, s_val = val_scores["y_true"].to_numpy(), val_scores["score"].to_numpy()
    scores = image_scores(best_weights, class_names, "test")
    scores.to_csv(ARTIFACTS_DIR / "test_scores.csv", index=False)
    y, s = scores["y_true"].to_numpy(), scores["score"].to_numpy()

    coco_seen = coco_zero_shot_findings()
    coco_seen.to_csv(ARTIFACTS_DIR / "coco_zero_shot_seen.csv", header=["detections"])
    print("What COCO zero-shot detects in fire images:")
    print(coco_seen.head(10).to_string() if len(coco_seen) else "(nothing)")

    ap50 = per_class.set_index("class")["ap50"]
    baselines = pd.DataFrame([
        summarise("finetuned yolo11s @0.25", s, y, 0.25),
        summarise("COCO yolo11s zero-shot", np.zeros(len(s)), y, 0.5),
        summarise("predict nothing", np.zeros(len(s)), y, 0.5),
    ])
    for name in class_names:
        baselines[f"ap50_{name}"] = [ap50.get(name, np.nan), 0.0, 0.0]
    baselines.to_csv(ARTIFACTS_DIR / "baselines.csv", index=False)
    print(baselines.to_string(index=False))

    # 6. Cost against threshold, swept on validation
    sweep = cost_sweep(s_val, y_val)
    sweep.to_csv(ARTIFACTS_DIR / "cost_sweep.csv", index=False)
    central = "cost_per_1000 10x (central)"
    best_row = sweep.loc[sweep[central].idxmin()]
    model_alone_threshold = float(best_row["threshold"])
    print(f"Cheapest single threshold on validation (central costs): {model_alone_threshold:.2f}")
    print(best_row.to_string())
    print("\nCheapest threshold under each FN scenario:")
    for label in FN_SCENARIOS:
        col = f"cost_per_1000 {label}"
        print(f"  {label:>28}: t={sweep.loc[sweep[col].idxmin(), 'threshold']:.2f}"
              f"  cost/1000={sweep[col].min():,.0f} EUR")
    plot_cost_curve(sweep, model_alone_threshold, ARTIFACTS_DIR / "cost_curve.png")

    # 7. Two thresholds, within the operators available, chosen on validation
    grid = pd.DataFrame([policy(s_val, y_val, lo, hi) for lo in THRESHOLDS for hi in THRESHOLDS if hi >= lo])
    feasible = grid[grid["reviewers_on_shift"] <= REVIEWERS_AVAILABLE]
    chosen = feasible.loc[feasible["cost_per_1000"].idxmin()]
    unconstrained = grid.loc[grid["cost_per_1000"].idxmin()]
    print("Chosen on validation (fits the operators available):")
    print(chosen.to_string())
    print("\nCheapest on validation ignoring staffing:")
    print(unconstrained.to_string())

    # Every strategy is then reported on test, with the thresholds fixed above.
    alternatives = pd.DataFrame([
        {"strategy": "do nothing (never alert)", **policy(s, y, 1.01, 1.01)},
        {"strategy": "a person checks every image", **policy(s, y, 0.0, 1.01)},
        {"strategy": f"model alone (t={model_alone_threshold:.2f})",
         **policy(s, y, model_alone_threshold, model_alone_threshold)},
        {"strategy": f"model + review ({chosen.t_low:.2f}-{chosen.t_high:.2f})",
         **policy(s, y, chosen.t_low, chosen.t_high)},
    ])
    alternatives.to_csv(ARTIFACTS_DIR / "alternatives.csv", index=False)
    print("\n" + alternatives.to_string(index=False))

    # 8. Everything the dashboard needs
    config = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "class_names": class_names,
        "positive_class": "image shows fire or smoke",
        "costs": {"fp": C_FP, "tp": C_TP, "tn": C_TN, "fn": C_FN, "fn_scenarios": FN_SCENARIOS},
        "base_rate": {
            "value": BASE_RATE, "cases_per_day": CASES_PER_DAY, "cameras": CAMERAS,
            "fires_per_year": FIRES_PER_YEAR, "cameras_per_fire": CAMERAS_PER_FIRE,
            "minutes_visible": MINUTES_VISIBLE,
        },
        "review": {
            "seconds": REVIEW_SECONDS, "cost_per_hour": REVIEWER_COST_PER_HOUR,
            "cost_per_case": REVIEW_COST, "reviewers_available": REVIEWERS_AVAILABLE,
        },
        "operating_point": {
            "model_alone_threshold": model_alone_threshold,
            "t_low": float(chosen.t_low), "t_high": float(chosen.t_high),
        },
        "training": {
            "base_model": BASE_MODEL, "epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ,
            "batch": BATCH, "device": torch.cuda.get_device_name(0) if DEVICE == 0 else "cpu",
            "seed": SEED, "smoke_test": SMOKE_TEST,
        },
        "detection_test": {"map50": float(test_metrics.box.map50), "map50_95": float(test_metrics.box.map)},
    }
    (ARTIFACTS_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote {sorted(p.name for p in ARTIFACTS_DIR.iterdir())}")
