"""
Tests for the prediction API endpoints.

Uses mocked ML models and mocked DB to run without trained artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def _make_registry():
    """Build a minimal mock registry that passes is_ready checks."""
    registry = MagicMock()
    registry.is_ready = True
    registry.feature_names = [
        "air_temp_k", "process_temp_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min", "temp_delta_k", "power_kw",
        "tool_wear_torque", "type_encoded", "temp_ratio", "speed_torque_ratio",
    ]

    xgb = MagicMock()
    xgb.predict_proba.return_value = np.array([[0.7, 0.3]])

    iso = MagicMock()
    iso.predict.return_value = np.array([1])
    iso.decision_function.return_value = np.array([0.15])

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 11))
    scaler.feature_names = registry.feature_names

    import shap
    shap_exp = MagicMock()
    shap_exp.shap_values.return_value = np.zeros((1, 11))
    shap_exp.expected_value = 0.1

    registry.xgboost = xgb
    registry.primary_model = xgb
    registry.anomaly_detector = iso
    registry.scaler = scaler
    registry.metrics = {}
    return registry


VALID_PAYLOAD = {
    "machine_id": "M-101",
    "machine_type": "M",
    "air_temp_k": 298.5,
    "process_temp_k": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_nm": 42.8,
    "tool_wear_min": 197.0,
}


@pytest_asyncio.fixture
async def api_client():
    registry = _make_registry()

    import app.ml.model_registry as mr_mod
    import app.ml.predictor as pred_mod

    from app.core.config import get_settings
    from app.ml.predictor import Predictor

    predictor = Predictor(registry, get_settings())

    with (
        patch.object(mr_mod, "_registry", registry),
        patch.object(pred_mod, "_predictor", predictor),
        patch("app.ml.model_registry.init_registry", return_value=registry),
        patch("app.ml.predictor.init_predictor", return_value=predictor),
        patch("app.db.mongodb.connect_db", new_callable=AsyncMock),
        patch("app.db.mongodb.close_db", new_callable=AsyncMock),
        patch("app.db.mongodb.get_db", return_value=MagicMock(
            __getitem__=MagicMock(return_value=MagicMock(
                insert_one=AsyncMock(),
                update_one=AsyncMock(),
            ))
        )),
        patch("app.db.seed.seed_machines", new_callable=AsyncMock),
    ):
        from app.main import app as fastapi_app
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as ac:
            yield ac


@pytest.mark.asyncio
async def test_predict_returns_200(api_client):
    response = await api_client.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_response_schema(api_client):
    response = await api_client.post("/api/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert "failure_probability" in data
    assert "risk_level" in data
    assert "prediction" in data
    assert "recommended_action" in data
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert data["prediction"] in {"NORMAL", "FAILURE_RISK"}
    assert 0.0 <= data["failure_probability"] <= 1.0


@pytest.mark.asyncio
async def test_predict_invalid_temp_below_range(api_client):
    bad_payload = {**VALID_PAYLOAD, "air_temp_k": 100.0}  # below 290 minimum
    response = await api_client.post("/api/predict", json=bad_payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_field(api_client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "torque_nm"}
    response = await api_client.post("/api/predict", json=incomplete)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_anomaly_endpoint(api_client):
    response = await api_client.post("/api/anomaly", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "anomaly_label" in data
    assert isinstance(data["is_anomaly"], bool)
    assert 0.0 <= data["anomaly_score"] <= 1.0


@pytest.mark.asyncio
async def test_invalid_machine_type(api_client):
    bad = {**VALID_PAYLOAD, "machine_type": "X"}  # not L/M/H
    response = await api_client.post("/api/predict", json=bad)
    assert response.status_code == 422
