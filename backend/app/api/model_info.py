"""
Model information endpoint.

GET /api/model/info — returns model metadata and evaluation metrics
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.ml.model_registry import get_registry
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/model/info",
    tags=["Model"],
    summary="Model metadata and evaluation metrics",
)
async def model_info():
    """
    Returns:
    - Primary model name
    - Feature names
    - Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
    - Which models are available
    """
    registry = get_registry()
    settings = get_settings()

    available = []
    if registry.xgboost:
        available.append("XGBoost")
    if registry.random_forest:
        available.append("Random Forest")
    if registry.logistic_regression:
        available.append("Logistic Regression")

    if not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run the training pipeline first.",
        )

    return {
        "primary_model": "XGBoost",
        "model_version": settings.model_version,
        "feature_count": len(registry.feature_names),
        "feature_names": registry.feature_names,
        "metrics": registry.metrics,
        "models_available": available,
        "is_ready": registry.is_ready,
        "risk_thresholds": {
            "LOW": f"< {settings.risk_low_threshold}",
            "MEDIUM": f"{settings.risk_low_threshold} – {settings.risk_medium_threshold}",
            "HIGH": f"{settings.risk_medium_threshold} – {settings.risk_high_threshold}",
            "CRITICAL": f">= {settings.risk_high_threshold}",
        },
    }
