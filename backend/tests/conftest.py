"""
Pytest configuration and shared fixtures.

Uses httpx.AsyncClient with the FastAPI app for async integration tests.
MongoDB is mocked to avoid requiring a live database during CI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Ensure project root is in path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def mock_registry():
    """Mock model registry — avoids needing trained .joblib files during tests."""
    import numpy as np

    registry = MagicMock()
    registry.is_ready = True
    registry.feature_names = [
        "air_temp_k", "process_temp_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min", "temp_delta_k", "power_kw",
        "tool_wear_torque", "type_encoded", "temp_ratio", "speed_torque_ratio",
    ]

    # Mock XGBoost: always returns ~0.3 failure probability
    xgb_mock = MagicMock()
    xgb_mock.predict_proba.return_value = np.array([[0.7, 0.3]])
    xgb_mock.feature_importances_ = np.ones(11) / 11
    registry.xgboost = xgb_mock
    registry.primary_model = xgb_mock

    # Mock RandomForest
    rf_mock = MagicMock()
    rf_mock.predict_proba.return_value = np.array([[0.65, 0.35]])
    registry.random_forest = rf_mock

    # Mock IsolationForest
    iso_mock = MagicMock()
    iso_mock.predict.return_value = np.array([1])  # Normal
    iso_mock.decision_function.return_value = np.array([0.15])
    registry.anomaly_detector = iso_mock

    # Mock scaler
    scaler_mock = MagicMock()
    scaler_mock.transform.return_value = np.zeros((1, 11))
    scaler_mock.feature_names = registry.feature_names
    registry.scaler = scaler_mock

    registry.metrics = {
        "test_set_size": 1500,
        "models": {
            "XGBoost": {
                "accuracy": 0.9812,
                "precision": 0.9234,
                "recall": 0.8956,
                "f1": 0.9093,
                "roc_auc": 0.9876,
                "pr_auc": 0.9542,
            }
        },
    }

    return registry


@pytest.fixture(scope="session")
def mock_predictor(mock_registry):
    """Instantiate a real Predictor with mocked registry."""
    from app.core.config import get_settings
    from app.ml.predictor import Predictor

    return Predictor(mock_registry, get_settings())


@pytest_asyncio.fixture
async def client(mock_registry, mock_predictor):
    """Async test client with mocked ML and DB layers."""
    from app.app.ml.model_registry import _registry
    import app.ml.model_registry as mr_module
    import app.ml.predictor as pred_module

    with (
        patch.object(mr_module, "_registry", mock_registry),
        patch.object(pred_module, "_predictor", mock_predictor),
        patch("app.db.mongodb.get_db", return_value=MagicMock()),
        patch("app.db.mongodb.connect_db", new_callable=AsyncMock),
        patch("app.db.mongodb.close_db", new_callable=AsyncMock),
        patch("app.db.seed.seed_machines", new_callable=AsyncMock),
    ):
        from app.main import app as fastapi_app
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as ac:
            yield ac


@pytest.fixture
def valid_sensor_payload():
    return {
        "machine_id": "M-101",
        "machine_type": "M",
        "air_temp_k": 298.5,
        "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551.0,
        "torque_nm": 42.8,
        "tool_wear_min": 197.0,
    }
