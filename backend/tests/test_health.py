"""Tests for the /api/health endpoint."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_health_returns_ok():
    """Health endpoint must return 200 with status=ok."""
    with (
        patch("app.ml.model_registry.init_registry", return_value=MagicMock(is_ready=False)),
        patch("app.ml.predictor.init_predictor", return_value=MagicMock()),
        patch("app.db.mongodb.connect_db", new_callable=AsyncMock),
        patch("app.db.mongodb.close_db", new_callable=AsyncMock),
        patch("app.db.seed.seed_machines", new_callable=AsyncMock),
    ):
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_has_service_name():
    with (
        patch("app.ml.model_registry.init_registry", return_value=MagicMock(is_ready=False)),
        patch("app.ml.predictor.init_predictor", return_value=MagicMock()),
        patch("app.db.mongodb.connect_db", new_callable=AsyncMock),
        patch("app.db.mongodb.close_db", new_callable=AsyncMock),
        patch("app.db.seed.seed_machines", new_callable=AsyncMock),
    ):
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/health")

    assert response.json()["service"] == "Predictive Maintenance AI"
