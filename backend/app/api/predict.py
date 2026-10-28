"""
Prediction API endpoints.

POST /api/predict               — failure probability + risk level
POST /api/anomaly               — anomaly detection only
POST /api/predict-and-analyze   — full analysis (prediction + anomaly + SHAP)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app.db.mongodb import get_db
from app.ml.predictor import get_predictor
from app.schemas.schemas import (
    AnomalyResponse,
    FullAnalysisResponse,
    PredictionResponse,
    SensorInput,
)
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


async def _store_prediction(db, machine_id: str | None, result: dict) -> None:
    """Persist prediction record to MongoDB."""
    if not machine_id:
        return
    doc = {
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k != "shap_explanation"},
        "model_version": get_settings().model_version,
    }
    await db["predictions"].insert_one(doc)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict machine failure probability",
)
async def predict(payload: SensorInput):
    """
    Accepts sensor readings and returns:
    - Failure probability (0–1)
    - Risk level (LOW / MEDIUM / HIGH / CRITICAL)
    - Recommended action
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(payload.model_dump())
        settings = get_settings()

        # Store in MongoDB (best-effort)
        try:
            db = get_db()
            await _store_prediction(db, payload.machine_id, result)
        except Exception as db_err:
            logger.warning("Failed to store prediction: %s", db_err)

        return PredictionResponse(
            machine_id=payload.machine_id,
            model_version=settings.model_version,
            **result,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/anomaly",
    response_model=AnomalyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Detect anomalous sensor patterns",
)
async def detect_anomaly(payload: SensorInput):
    """
    Runs Isolation Forest anomaly detection.

    Note: Anomaly detection is distinct from failure prediction.
    A reading can be anomalous without imminent failure, and vice versa.
    """
    try:
        predictor = get_predictor()
        result = predictor.detect_anomaly(payload.model_dump())
        return AnomalyResponse(machine_id=payload.machine_id, **result)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Anomaly detection error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/predict-and-analyze",
    response_model=FullAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Full analysis: prediction + anomaly detection + SHAP explanation",
)
async def predict_and_analyze(payload: SensorInput):
    """
    Runs the complete analysis pipeline:
    1. Failure probability (XGBoost)
    2. Anomaly detection (Isolation Forest)
    3. SHAP feature explanation

    This is the recommended endpoint for the dashboard.
    """
    try:
        predictor = get_predictor()
        settings = get_settings()
        result = predictor.predict_and_analyze(payload.model_dump())

        # Store in MongoDB (best-effort)
        try:
            db = get_db()
            await _store_prediction(db, payload.machine_id, result)
            # Update machine status in DB
            if payload.machine_id:
                status_map = {"LOW": "healthy", "MEDIUM": "warning", "HIGH": "warning", "CRITICAL": "critical"}
                new_status = status_map.get(result["risk_level"], "healthy")
                await db["machines"].update_one(
                    {"machine_id": payload.machine_id},
                    {"$set": {"status": new_status, "last_prediction": datetime.now(timezone.utc).isoformat()}},
                )
        except Exception as db_err:
            logger.warning("DB update failed: %s", db_err)

        return FullAnalysisResponse(
            machine_id=payload.machine_id,
            model_version=settings.model_version,
            **result,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Full analysis error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
