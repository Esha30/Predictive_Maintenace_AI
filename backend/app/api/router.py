"""Central API router — registers all sub-routers."""

from fastapi import APIRouter

from app.api import health, predict, machines, predictions, analytics, model_info

router = APIRouter(prefix="/api")

router.include_router(health.router)
router.include_router(predict.router)
router.include_router(machines.router)
router.include_router(predictions.router)
router.include_router(analytics.router)
router.include_router(model_info.router)
