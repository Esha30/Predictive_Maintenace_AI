"""
Core prediction logic — bridges raw sensor input → ML inference → structured output.

This module is the single place where:
  1. Raw SensorInput is converted to a feature-engineered DataFrame
  2. The scaler transforms it to the same space seen during training
  3. The XGBoost classifier predicts failure probability
  4. The IsolationForest detects anomalies
  5. SHAP explains the prediction
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure ml.* is importable
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.preprocessing.feature_engineering import engineer_features, rename_columns


class Predictor:
    """
    End-to-end inference pipeline.

    Usage:
        predictor = Predictor(registry, settings)
        result = predictor.predict_and_analyze(sensor_input_dict)
    """

    def __init__(self, registry, settings) -> None:
        self.registry = registry
        self.settings = settings

    # ── Raw input → engineered DataFrame ───────────────────────
    def _prepare_features(self, sensor_data: dict) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Convert raw API input into a scaled feature array.

        Parameters
        ----------
        sensor_data : dict
            Keys matching the AI4I raw column names (already cleaned):
            air_temp_k, process_temp_k, rotational_speed_rpm,
            torque_nm, tool_wear_min, type (L/M/H)

        Returns
        -------
        (df_engineered, X_scaled)
        """
        # Build a one-row DataFrame with raw sensor names
        raw = {
            "UDI": 0,
            "Product ID": "M000",
            "Type": sensor_data.get("machine_type", "M"),
            "Air temperature [K]": sensor_data["air_temp_k"],
            "Process temperature [K]": sensor_data["process_temp_k"],
            "Rotational speed [rpm]": sensor_data["rotational_speed_rpm"],
            "Torque [Nm]": sensor_data["torque_nm"],
            "Tool wear [min]": sensor_data["tool_wear_min"],
            "Machine failure": 0,
            "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        }
        df = pd.DataFrame([raw])
        df = rename_columns(df)
        df = engineer_features(df)
        X_scaled = self.registry.scaler.transform(df)
        return df, X_scaled

    # ── Failure prediction ──────────────────────────────────────
    def predict(self, sensor_data: dict) -> dict[str, Any]:
        """
        Predict failure probability using the primary (XGBoost) model.

        Returns dict with:
          failure_probability, risk_level, prediction, recommended_action
        """
        if not self.registry.is_ready:
            raise RuntimeError("Models are not loaded. Run the training pipeline first.")

        _, X = self._prepare_features(sensor_data)
        model = self.registry.primary_model

        prob = float(model.predict_proba(X)[0, 1])
        risk = self.settings.classify_risk(prob)
        prediction = "FAILURE_RISK" if prob >= self.settings.risk_low_threshold else "NORMAL"

        return {
            "failure_probability": round(prob, 4),
            "risk_level": risk,
            "prediction": prediction,
            "recommended_action": self.settings.recommended_action(risk),
        }

    # ── Anomaly detection ───────────────────────────────────────
    def detect_anomaly(self, sensor_data: dict) -> dict[str, Any]:
        """
        Detect anomalous sensor patterns using Isolation Forest.

        Returns dict with:
          is_anomaly (bool), anomaly_score (float, normalised 0–1),
          anomaly_label ("Normal" | "Anomaly Detected")
        """
        if not self.registry.anomaly_detector:
            raise RuntimeError("Anomaly detector not loaded.")

        _, X = self._prepare_features(sensor_data)
        detector = self.registry.anomaly_detector

        # IsolationForest: predict returns 1 (normal) or -1 (anomaly)
        pred = int(detector.predict(X)[0])
        # decision_function: lower = more anomalous
        raw_score = float(detector.decision_function(X)[0])
        # Normalise to [0, 1] where 1 = most anomalous
        anomaly_score = float(1 / (1 + np.exp(raw_score * 10)))  # sigmoid

        return {
            "is_anomaly": pred == -1,
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_label": "Anomaly Detected" if pred == -1 else "Normal",
        }

    # ── SHAP explanation ────────────────────────────────────────
    def explain(self, sensor_data: dict) -> dict[str, Any]:
        """
        Generate SHAP explanations for the XGBoost prediction.

        Returns dict with top contributing features and all SHAP values.
        """
        import shap

        if not self.registry.xgboost:
            raise RuntimeError("XGBoost model not loaded — cannot explain.")

        _, X = self._prepare_features(sensor_data)

        explainer = shap.TreeExplainer(self.registry.xgboost)
        shap_values = explainer.shap_values(X)

        # Handle both list and array returns (shap version differences)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1
        else:
            sv = shap_values[0]

        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = float(explainer.expected_value[1])

        feature_names = self.registry.feature_names
        contributions = [
            {
                "feature": name,
                "feature_value": round(float(X[0, i]), 4),
                "shap_value": round(float(sv[i]), 4),
                "direction": "positive" if sv[i] > 0 else "negative",
            }
            for i, name in enumerate(feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(base_value, 4),
            "top_features": contributions[:6],
            "all_contributions": contributions,
        }

    # ── Combined full analysis ──────────────────────────────────
    def predict_and_analyze(self, sensor_data: dict) -> dict[str, Any]:
        """Run prediction + anomaly detection + SHAP in one call."""
        pred = self.predict(sensor_data)
        anomaly = self.detect_anomaly(sensor_data)
        explanation = self.explain(sensor_data)

        return {
            **pred,
            "anomaly_score": anomaly["anomaly_score"],
            "is_anomaly": anomaly["is_anomaly"],
            "anomaly_label": anomaly["anomaly_label"],
            "shap_explanation": explanation,
        }


# Module-level singleton
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialised. Call init_predictor() first.")
    return _predictor


def init_predictor(registry, settings) -> Predictor:
    global _predictor
    _predictor = Predictor(registry, settings)
    return _predictor
