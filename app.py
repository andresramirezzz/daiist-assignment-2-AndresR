"""
Assignment 2: Gradio dashboard for the Madrid forest-fire camera alerts.

Run via `uv run python main.py app`. It only reads what train.py produced:
the committed weights in weights/ and the evaluation outputs in artifacts/.
Nothing is trained here. The thresholds were chosen on the validation split;
the outcomes shown here are on the test split.
"""

import json
import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

CONFIG = json.loads((ARTIFACTS / "config.json").read_text())
SCORES = pd.read_csv(ARTIFACTS / "test_scores.csv")
SWEEP = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
PER_CLASS = pd.read_csv(ARTIFACTS / "per_class.csv")
BASELINES = pd.read_csv(ARTIFACTS / "baselines.csv")
ALTERNATIVES = pd.read_csv(ARTIFACTS / "alternatives.csv")
DETECTION_CM = json.loads((ARTIFACTS / "detection_confusion.json").read_text())

COSTS = CONFIG["costs"]
BASE_RATE = CONFIG["base_rate"]["value"]
CASES_PER_DAY = CONFIG["base_rate"]["cases_per_day"]
REVIEW = CONFIG["review"]
OPERATING = CONFIG["operating_point"]
CENTRAL = "cost_per_1000 10x (central)"

MODEL = YOLO(str(WEIGHTS))
Y = SCORES["y_true"].to_numpy()
S = SCORES["score"].to_numpy()


def band_rates(score: np.ndarray, t_low: float, t_high: float) -> tuple[float, float, float]:
    """Share of cases that pass, go to review, and alert. Mirrors train.py."""
    alert = (score >= t_high).mean()
    passed = (score < t_low).mean()
    return passed, 1 - alert - passed, alert


def upper_rate(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate seen k times in n. Mirrors train.py."""
    def at_most_k(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at_most_k(mid) > 1 - confidence else (lo, mid)
    return hi


def evaluate_policy(t_low: float, t_high: float):
    t_high = max(t_high, t_low)
    p_pass, p_rev, p_alert = band_rates(S[Y == 1], t_low, t_high)
    n_pass, n_rev, n_alert = band_rates(S[Y == 0], t_low, t_high)
    cost = (
        BASE_RATE * (p_alert * COSTS["tp"] + p_rev * (COSTS["tp"] + REVIEW["cost_per_case"]) + p_pass * COSTS["fn"])
        + (1 - BASE_RATE) * (n_alert * COSTS["fp"] + n_rev * (COSTS["tn"] + REVIEW["cost_per_case"]) + n_pass * COSTS["tn"])
    )
    review_share = BASE_RATE * p_rev + (1 - BASE_RATE) * n_rev
    reviews_per_day = review_share * CASES_PER_DAY
    reviewers = reviews_per_day * REVIEW["seconds"] / 3600 / 24

    def counts(mask):
        sc = S[mask]
        return [int((sc < t_low).sum()), int(((sc >= t_low) & (sc < t_high)).sum()), int((sc >= t_high).sum())]

    test_cm = pd.DataFrame(
        [counts(Y == 1), counts(Y == 0)],
        index=["true: fire", "true: no fire"],
        columns=["pass", "human review", "alert"],
    ).reset_index(names="test split (images)")
    per_1000 = pd.DataFrame(
        [
            [1000 * BASE_RATE * p_pass, 1000 * BASE_RATE * p_rev, 1000 * BASE_RATE * p_alert],
            [1000 * (1 - BASE_RATE) * n_pass, 1000 * (1 - BASE_RATE) * n_rev, 1000 * (1 - BASE_RATE) * n_alert],
        ],
        index=["true: fire", "true: no fire"],
        columns=["pass", "human review", "alert"],
    ).round(3).reset_index(names="per 1,000 production images")

    negatives = int((Y == 0).sum())
    false_alerts = (1 - BASE_RATE) * n_alert * CASES_PER_DAY
    false_alerts_upper = (1 - BASE_RATE) * upper_rate(int((S[Y == 0] >= t_high).sum()), negatives) * CASES_PER_DAY

    fits = "fits" if reviewers <= REVIEW["reviewers_available"] else "does NOT fit"
    summary = (
        f"### Expected cost: **{1000 * cost:,.1f} EUR per 1,000 images** ({cost * CASES_PER_DAY:,.0f} EUR/day)\n"
        f"- Sent to human review: **{100 * review_share:.2f}%** of images, {reviews_per_day:,.0f} a day\n"
        f"- Operators needed on shift (24 h, {REVIEW['seconds']} s per check): **{reviewers:.2f}**, "
        f"which {fits} the {REVIEW['reviewers_available']} available\n"
        f"- Fires that pass unseen: **{100 * p_pass:.1f}%**; "
        f"false alerts a day: **{false_alerts:,.1f}**\n"
        f"- With only {negatives} non-fire test images, the false-alarm rate could still be higher: "
        f"95% upper bound **{false_alerts_upper:,.0f} false alerts a day**"
    )
    return summary, test_cm, per_1000, cost_curve(t_low, t_high)


def cost_curve(t_low: float | None = None, t_high: float | None = None) -> go.Figure:
    fig = go.Figure()
    for column in [c for c in SWEEP.columns if c.startswith("cost_per_1000")]:
        is_central = column == CENTRAL
        fig.add_trace(go.Scatter(
            x=SWEEP["threshold"], y=SWEEP[column], mode="lines",
            name=column.replace("cost_per_1000 ", "FN cost "),
            line={"width": 3 if is_central else 1}, opacity=1 if is_central else 0.6,
        ))
    chosen = OPERATING["model_alone_threshold"]
    fig.add_vline(x=chosen, line_dash="dash", line_color="black",
                  annotation_text=f"chosen t={chosen:.2f}", annotation_position="top")
    if t_low is not None:
        fig.add_vrect(x0=t_low, x1=max(t_high, t_low), fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="review band", annotation_position="bottom left")
    fig.update_layout(
        title="Validation split (where the thresholds were chosen)",
        xaxis_title="alert threshold (model alone)", yaxis_title="expected cost per 1,000 images (EUR)",
        yaxis_type="log", height=420, margin={"t": 40}, legend={"font": {"size": 10}},
    )
    return fig


def predict(image, t_low: float, t_high: float):
    if image is None:
        return None, "Upload an image."
    result = MODEL.predict(image, conf=min(t_low, t_high), verbose=False)[0]
    score = float(result.boxes.conf.max()) if len(result.boxes) else 0.0
    if score >= t_high:
        verdict = "ALERT: send to Madrid 112"
    elif score >= t_low:
        verdict = "HUMAN REVIEW: operator checks the feed"
    else:
        verdict = "PASS"
    detections = ", ".join(
        f"{MODEL.names[int(c)]} {float(p):.2f}" for c, p in list(zip(result.boxes.cls, result.boxes.conf))[:10]
    ) or "none"
    if len(result.boxes) > 10:
        detections += f" (+{len(result.boxes) - 10} more)"
    return result.plot()[..., ::-1], f"### {verdict}\nImage score: {score:.2f}. Detections: {detections}"


def detection_confusion_table() -> pd.DataFrame:
    labels = DETECTION_CM["labels"]
    return pd.DataFrame(
        DETECTION_CM["matrix"], index=[f"predicted {l}" for l in labels], columns=[f"true {l}" for l in labels]
    ).reset_index(names="")


with gr.Blocks(title="Madrid forest-fire camera alerts") as demo:
    gr.Markdown(
        "# Madrid forest-fire camera alerts\n"
        f"{CONFIG['base_rate']['cameras']} tower cameras, one image a minute, 24 h "
        f"({CASES_PER_DAY:,} images a day). Assumed base rate {BASE_RATE:.5f} "
        f"(1 in {1 / BASE_RATE:,.0f}). Costs: FP {COSTS['fp']:,} EUR, FN {COSTS['fn']:,} EUR, "
        f"TP {COSTS['tp']:,} EUR, review {REVIEW['cost_per_case']:.2f} EUR."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=OPERATING["t_low"], step=0.01, label="t_low: below this, pass")
        t_high = gr.Slider(0, 1, value=OPERATING["t_high"], step=0.01, label="t_high: at or above this, alert")

    with gr.Tab("Operating policy"):
        summary = gr.Markdown()
        with gr.Row():
            test_cm = gr.Dataframe(label="Outcomes on the test split")
            prod_cm = gr.Dataframe(label="Expected outcomes per 1,000 production images")
        curve = gr.Plot(label="Cost against threshold")
        gr.Markdown("#### Strategies compared at the chosen operating point (test split)")
        gr.Dataframe(ALTERNATIVES.round(3))

    with gr.Tab("Try an image"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="Camera image")
            annotated = gr.Image(label="Detections")
        verdict = gr.Markdown()
        upload.change(predict, [upload, t_low, t_high], [annotated, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown(
            f"Test split: mAP50 **{CONFIG['detection_test']['map50']:.3f}**, "
            f"mAP50-95 **{CONFIG['detection_test']['map50_95']:.3f}**. "
            f"Positive class for the decision: {CONFIG['positive_class']}."
        )
        gr.Dataframe(PER_CLASS.round(3), label="Per-class detection metrics")
        gr.Dataframe(detection_confusion_table(), label="Detection confusion matrix (conf 0.25)")
        gr.Dataframe(BASELINES.round(3), label="Finetuned model against baselines (image level)")

    for slider in (t_low, t_high):
        slider.change(evaluate_policy, [t_low, t_high], [summary, test_cm, prod_cm, curve])
        slider.change(predict, [upload, t_low, t_high], [annotated, verdict])
    demo.load(evaluate_policy, [t_low, t_high], [summary, test_cm, prod_cm, curve])


if __name__ == "__main__":
    demo.launch()
