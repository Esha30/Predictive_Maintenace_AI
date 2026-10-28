"""
Model Registry — loads and caches all trained ML artifacts at startup.

Models loaded:
  - scaler.joblib          : fitted StandardScaler (DataPreprocessor)
  - xgboost.joblib         : primary classifier (XGBClassifier)
  - random_forest.joblib   : baseline ensemble
  - logistic_regression.joblib : baseline linear
  - anomaly_detector.joblib : IsolationForest
  - metrics.json           : evaluation metrics from test set

All models are loaded once and shared across requests (singleton pattern).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# Add project root to path so ml.* imports work from the backend
_ROOT = Path(__file__).parent.parent.parent.parent  # Predictive_Maintenance_AI/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class ModelRegistry:
    """Singleton container for all trained model artifacts."""

    _instance: "ModelRegistry | None" = None

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.scaler = None
        self.xgboost = None
        self.random_forest = None
        self.logistic_regression = None
        self.anomaly_detector = None
        self.metrics: dict = {}
        self.feature_names: list[str] = []
        self._loaded = False

    def load(self) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            return

        logger.info("Loading ML models from %s …", self.model_dir)

        # Scaler (DataPreprocessor wrapper)
        scaler_path = self.model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.feature_names = self.scaler.feature_names
            logger.info("✓ Scaler loaded — features: %s", self.feature_names)
        else:
            logger.warning("scaler.joblib not found — models not trained yet.")

        # Classifiers
        for attr, filename in [
            ("xgboost", "xgboost"),
            ("random_forest", "random_forest"),
            ("logistic_regression", "logistic_regression"),
        ]:
            path = self.model_dir / f"{filename}.joblib"
            if path.exists():
                setattr(self, attr, joblib.load(path))
                logger.info("✓ %s loaded", filename)
            else:
                logger.warning("%s not found", path)

        # Anomaly detector
        anomaly_path = self.model_dir / "anomaly_detector.joblib"
        if anomaly_path.exists():
            self.anomaly_detector = joblib.load(anomaly_path)
            logger.info("✓ Anomaly detector loaded")

        # Evaluation metrics
        metrics_path = self.model_dir.parent / "evaluation" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)
            logger.info("✓ Metrics loaded from %s", metrics_path)

        self._loaded = True
        logger.info("Model registry initialisation complete.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.xgboost is not None and self.scaler is not None

    @property
    def primary_model(self):
        """Return the best model (XGBoost preferred)."""
        return self.xgboost or self.random_forest or self.logistic_regression


# Module-level singleton
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialised. Call init_registry() first.")
    return _registry


def init_registry(model_dir: Path) -> ModelRegistry:
    global _registry
    _registry = ModelRegistry(model_dir)
    _registry.load()
    return _registry
