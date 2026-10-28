"""
FastAPI application entry point.

Startup sequence:
  1. Logging is configured
  2. ML models are loaded from disk (or training is triggered if missing)
  3. MongoDB connection is established
  4. Database is seeded with sample machines (first run only)
  5. All API routes are mounted
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.api.router import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown lifecycle."""
    settings = get_settings()

    # 1. Logging
    setup_logging(settings.log_level)
    logger.info("Starting Predictive Maintenance AI v%s", settings.model_version)

    # 2. Load ML models
    from app.ml.model_registry import init_registry
    from app.ml.predictor import init_predictor

    model_dir = Path(settings.model_path)
    registry = init_registry(model_dir)

    if not registry.is_ready:
        logger.warning(
            "⚠️  Models not found at %s. "
            "Run 'python -m ml.training.train && python -m ml.training.anomaly' first.",
            model_dir,
        )
    else:
        logger.info("✅ All ML models loaded successfully.")

    init_predictor(registry, settings)

    # 3. Connect to MongoDB
    from app.db.mongodb import connect_db
    try:
        await connect_db()

        # 4. Seed database
        from app.db.seed import seed_machines
        await seed_machines()
    except Exception as exc:
        logger.warning("MongoDB not available: %s — running without persistence.", exc)

    logger.info("🚀 API ready at http://%s:%d", settings.api_host, settings.api_port)

    yield  # Application runs here

    # Shutdown
    from app.db.mongodb import close_db
    await close_db()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Predictive Maintenance AI",
        description=(
            "Production-grade AI system for industrial machine failure prediction. "
            "Uses XGBoost, Isolation Forest, and SHAP for end-to-end predictive maintenance."
        ),
        version=settings.model_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    return app


app = create_app()
