"""
Model evaluation script.

Evaluates all three trained models on the held-out test set and writes
results to ml/evaluation/metrics.json.

Generated artifacts:
  ml/evaluation/metrics.json          — all metrics for all models
  ml/evaluation/confusion_matrix.png  — XGBoost confusion matrix
  ml/evaluation/roc_curve.png         — ROC curves for all models
  ml/evaluation/feature_importance.png — XGBoost feature importances

IMPORTANT: This script NEVER touches training or validation data.
All metrics are computed exclusively on the test set (X_test.npy, y_test.npy).

Usage:
  python -m ml.evaluation.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR   = ROOT / "ml" / "models"
EVAL_DIR     = ROOT / "ml" / "evaluation"
METRICS_PATH = EVAL_DIR / "metrics.json"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)),  4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)),        4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)),                   4),
        "pr_auc":    round(float(average_precision_score(y_true, y_proba)),         4),
    }


# ─────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Failure"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved confusion_matrix.png")


def plot_roc_curves(models_data: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#6366f1", "#22c55e", "#f59e0b"]
    for i, entry in enumerate(models_data):
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_proba"])
        auc = entry["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{entry['name']} (AUC={auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print("  Saved roc_curve.png")


def plot_feature_importance(xgb_model, feature_names: list[str]) -> None:
    importances = xgb_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    ax.barh(
        [feature_names[i] for i in idx],
        [importances[i] for i in idx],
        color=colors[::-1],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importances", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("  Saved feature_importance.png")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Model Evaluation — Test Set")
    print("=" * 60)

    # Load test data
    X_test = np.load(MODELS_DIR / "X_test.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")
    print(f"Test set: {X_test.shape[0]} samples, positive rate: {y_test.mean():.3%}")

    # Load feature names
    from ml.preprocessing.feature_engineering import FEATURE_COLUMNS

    # Load models
    model_files = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }

    all_metrics = {}
    models_data = []

    for name, fname in model_files.items():
        path = MODELS_DIR / f"{fname}.joblib"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue

        model = joblib.load(path)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        all_metrics[name] = metrics
        models_data.append({"name": name, "y_true": y_test, "y_proba": y_proba, "metrics": metrics})

        print(f"\n  {name}")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

    # XGBoost confusion matrix + feature importance
    if "XGBoost" in all_metrics:
        xgb_entry = next(e for e in models_data if e["name"] == "XGBoost")
        xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
        y_pred_xgb = xgb_model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
        plot_feature_importance(xgb_model, FEATURE_COLUMNS)

    # ROC curves (all models together)
    plot_roc_curves(models_data)

    # Save metrics JSON
    output = {
        "test_set_size": int(X_test.shape[0]),
        "positive_rate": float(y_test.mean()),
        "models": all_metrics,
        "best_model": "XGBoost",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
