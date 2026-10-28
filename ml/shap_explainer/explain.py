"""
SHAP explainability module.

Generates SHAP explanations for XGBoost predictions.

Usage:
  from ml.shap_explainer.explain import SHAPExplainer
  explainer = SHAPExplainer()
  result = explainer.explain_instance(X_single_row)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing.feature_engineering import FEATURE_COLUMNS

MODELS_DIR = ROOT / "ml" / "models"


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for XGBoost model.

    Provides per-instance feature contribution values.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else MODELS_DIR / "xgboost.joblib"
        self.model = joblib.load(path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = FEATURE_COLUMNS

    def explain_instance(self, X: np.ndarray) -> dict[str, Any]:
        """
        Compute SHAP values for a single instance.

        Parameters
        ----------
        X : np.ndarray, shape (1, n_features)
            Preprocessed (scaled) feature vector.

        Returns
        -------
        dict with:
          base_value        : float  — model output at baseline
          prediction_value  : float  — model's predicted probability
          top_features      : list of {name, value, shap_value, direction}
          all_shap_values   : list of floats (one per feature)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values may be (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 SHAP values
        else:
            sv = shap_values[0]

        base_value = float(self.explainer.expected_value)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = float(self.explainer.expected_value[1])

        # Build sorted feature contributions
        contributions = [
            {
                "feature": name,
                "feature_value": float(X[0, i]),
                "shap_value": float(sv[i]),
                "direction": "positive" if sv[i] > 0 else "negative",
            }
            for i, name in enumerate(self.feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": base_value,
            "top_features": contributions[:5],
            "all_contributions": contributions,
            "all_shap_values": sv.tolist(),
        }

    def generate_summary_plot(self, X: np.ndarray, output_path: str | Path) -> None:
        """Generate and save a SHAP summary plot (beeswarm)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        shap.summary_plot(
            sv, X,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved → {output_path}")
