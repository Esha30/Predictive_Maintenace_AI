"""
Prediction history endpoints.

GET /api/predictions                — recent predictions (all machines)
GET /api/predictions/{machine_id}   — predictions for a specific machine
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.db.mongodb import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


def _clean(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc


@router.get(
    "/predictions",
    tags=["Predictions"],
    summary="List recent predictions",
)
async def list_predictions(
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    machine_id: Optional[str] = Query(None, description="Filter by machine ID"),
):
    try:
        db = get_db()
        query = {}
        if machine_id:
            query["machine_id"] = machine_id

        cursor = db["predictions"].find(query).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_clean(d) for d in docs]
    except Exception as e:
        return []


@router.get(
    "/predictions/{machine_id}",
    tags=["Predictions"],
    summary="Get prediction history for a machine",
)
async def get_machine_predictions(
    machine_id: str,
    limit: int = Query(30, ge=1, le=100),
):
    try:
        db = get_db()
        # Verify machine exists
        machine = await db["machines"].find_one({"machine_id": machine_id})
        if not machine:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Machine '{machine_id}' not found.",
            )

        cursor = (
            db["predictions"]
            .find({"machine_id": machine_id})
            .sort("timestamp", -1)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [_clean(d) for d in docs]
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"MongoDB unavailable for machine predictions: {e}")
        return []
