"""
evaluate.py

Model monitoring / performance evaluation — the metrics an interviewer
(or an MLOps process) cares about, even though a business stakeholder
mostly cares about dollars saved. Covers:
  - ROC-AUC, PR-AUC
  - Precision, recall, F1 at the default 0.5 threshold
  - ROC curve and Precision-Recall curve
  - Calibration curve: does a customer with "70% predicted churn
    probability" actually churn about 70% of the time? A model can have
    good ranking ability (AUC) while still being poorly calibrated, which
    matters a lot here since the ROI math directly multiplies churn_prob
    by CLV — a miscalibrated probability distorts every downstream dollar
    figure, not just the ranking.

Reads from data/test_predictions.csv (written by train.py), which has
both churn_prob and actual_churn for the held-out test set.

Run:
    python src/evaluate.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve

IN_PATH = "data/test_predictions.csv"
CHART_OUT = "data/model_evaluation.png"
METRICS_OUT = "data/model_metrics.csv"

DEFAULT_THRESHOLD = 0.5


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def plot_evaluation_charts(y_true: np.ndarray, y_prob: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.3f}")
    baseline = y_true.mean()
    axes[1].axhline(baseline, linestyle="--", color="gray", label=f"Baseline (churn rate = {baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    axes[2].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axes[2].set_xlabel("Predicted churn probability")
    axes[2].set_ylabel("Actual churn rate")
    axes[2].set_title("Calibration Curve")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_OUT, dpi=150)
    plt.close()
    print(f"Evaluation charts saved to {CHART_OUT}")


def main():
    df = pd.read_csv(IN_PATH)
    if "actual_churn" not in df.columns or "churn_prob" not in df.columns:
        raise ValueError(
            f"{IN_PATH} must have 'actual_churn' and 'churn_prob' columns "
            "(these are written by train.py — run that first)."
        )

    y_true = df["actual_churn"].values
    y_prob = df["churn_prob"].values

    metrics = compute_metrics(y_true, y_prob)
    print("=== Model Performance Metrics (test set) ===")
    for k, v in metrics.items():
        if k == "threshold":
            print(f"  Threshold used: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(y_true, (y_prob >= DEFAULT_THRESHOLD).astype(int))
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"                predicted_no  predicted_yes")
    print(f"  actual_no     {cm[0, 0]:>12}  {cm[0, 1]:>13}")
    print(f"  actual_yes    {cm[1, 0]:>12}  {cm[1, 1]:>13}")

    plot_evaluation_charts(y_true, y_prob)

    pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)
    print(f"\nMetrics saved to {METRICS_OUT}")


if __name__ == "__main__":
    main()