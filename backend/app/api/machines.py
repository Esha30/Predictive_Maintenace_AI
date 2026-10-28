"""
Machine management endpoints.

GET  /api/machines            — list all machines
POST /api/machines            — create a machine
GET  /api/machines/{id}       — get machine details
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.db.mongodb import get_db
from app.schemas.schemas import MachineCreate, MachineSchema

router = APIRouter()
logger = logging.getLogger(__name__)

# Fallback demo machines shown when MongoDB is unavailable
_DEMO_MACHINES = [
    {
        "machine_id": "M-1001",
        "machine_name": "CNC Lathe Alpha",
        "machine_type": "H",
        "status": "healthy",
        "installation_date": "2024-01-15",
        "location": "Floor A - Bay 1",
        "operating_hours": 12450,
        "last_maintenance": "2026-09-01T00:00:00Z",
    },
    {
        "machine_id": "M-1002",
        "machine_name": "Hydraulic Press Beta",
        "machine_type": "M",
        "status": "warning",
        "installation_date": "2023-06-20",
        "location": "Floor A - Bay 3",
        "operating_hours": 18230,
        "last_maintenance": "2026-08-15T00:00:00Z",
    },
    {
        "machine_id": "M-1003",
        "machine_name": "Conveyor Unit Gamma",
        "machine_type": "L",
        "status": "healthy",
        "installation_date": "2025-03-10",
        "location": "Floor B - Bay 2",
        "operating_hours": 5120,
        "last_maintenance": "2026-09-10T00:00:00Z",
    },
    {
        "machine_id": "M-1004",
        "machine_name": "Welding Robot Delta",
        "machine_type": "H",
        "status": "critical",
        "installation_date": "2022-11-05",
        "location": "Floor C - Bay 1",
        "operating_hours": 24800,
        "last_maintenance": "2026-07-20T00:00:00Z",
    },
    {
        "machine_id": "M-1005",
        "machine_name": "Milling Machine Epsilon",
        "machine_type": "M",
        "status": "healthy",
        "installation_date": "2024-09-01",
        "location": "Floor A - Bay 5",
        "operating_hours": 8900,
        "last_maintenance": "2026-09-18T00:00:00Z",
    },
]


def _clean(doc: dict) -> dict:
    """Remove MongoDB _id field from document."""
    doc.pop("_id", None)
    return doc


@router.get(
    "/machines",
    response_model=List[MachineSchema],
    tags=["Machines"],
    summary="List all machines",
)
async def list_machines():
    try:
        db = get_db()
        machines = await db["machines"].find({}).to_list(length=200)
        return [_clean(m) for m in machines]
    except Exception as e:
        logger.warning(f"MongoDB unavailable for machines: {e}")
        return _DEMO_MACHINES


@router.post(
    "/machines",
    response_model=MachineSchema,
    status_code=status.HTTP_201_CREATED,
    tags=["Machines"],
    summary="Register a new machine",
)
async def create_machine(payload: MachineCreate):
    try:
        db = get_db()
        # Check for duplicate
        existing = await db["machines"].find_one({"machine_id": payload.machine_id})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Machine '{payload.machine_id}' already exists.",
            )
        doc = payload.model_dump()
        await db["machines"].insert_one(doc)
        return _clean(doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"MongoDB unavailable for create: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable. Cannot create machine.",
        )


@router.get(
    "/machines/{machine_id}",
    response_model=MachineSchema,
    tags=["Machines"],
    summary="Get machine details",
)
async def get_machine(machine_id: str):
    try:
        db = get_db()
        machine = await db["machines"].find_one({"machine_id": machine_id})
        if not machine:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Machine '{machine_id}' not found.",
            )
        return _clean(machine)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"MongoDB unavailable for get_machine: {e}")
        # Return a matching demo machine if available
        for m in _DEMO_MACHINES:
            if m["machine_id"] == machine_id:
                return m
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine '{machine_id}' not found.",
        )
