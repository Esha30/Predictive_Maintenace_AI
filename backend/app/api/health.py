"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Returns service health status.

    Used by Docker health checks and monitoring systems.
    """
    return {
        "status": "ok",
        "service": "Predictive Maintenance AI",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }
