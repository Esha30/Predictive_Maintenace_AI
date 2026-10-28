"""
Seed the database with 10 realistic sample machines.

Run once on startup if the machines collection is empty.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone, timedelta

from app.db.mongodb import get_db

logger = logging.getLogger(__name__)

MACHINE_TYPES = ["CNC Mill", "Hydraulic Press", "Conveyor Belt", "Pump", "Compressor"]
LOCATIONS = ["Plant A - Floor 1", "Plant A - Floor 2", "Plant B - Floor 1", "Plant B - Floor 2", "Warehouse C"]
STATUSES = ["healthy", "healthy", "healthy", "healthy", "warning", "warning", "critical"]  # weighted

SAMPLE_MACHINES = [
    {
        "machine_id": f"M-{100 + i}",
        "machine_name": f"{MACHINE_TYPES[i % len(MACHINE_TYPES)]} #{i + 1}",
        "machine_type": MACHINE_TYPES[i % len(MACHINE_TYPES)],
        "installation_date": (datetime.now(timezone.utc) - timedelta(days=random.randint(180, 1800))).isoformat(),
        "location": LOCATIONS[i % len(LOCATIONS)],
        "status": STATUSES[i % len(STATUSES)],
        "operating_hours": random.randint(500, 15000),
        "last_maintenance": (datetime.now(timezone.utc) - timedelta(days=random.randint(10, 90))).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for i in range(10)
]


async def seed_machines() -> None:
    """Insert sample machines if the collection is empty."""
    db = get_db()
    count = await db["machines"].count_documents({})
    if count > 0:
        logger.info("Machines collection already has %d documents — skipping seed.", count)
        return

    # Ensure unique machine_id index
    await db["machines"].create_index("machine_id", unique=True)
    await db["predictions"].create_index([("machine_id", 1), ("timestamp", -1)])
    await db["sensor_readings"].create_index([("machine_id", 1), ("timestamp", -1)])

    result = await db["machines"].insert_many(SAMPLE_MACHINES)
    logger.info("Seeded %d machines into the database.", len(result.inserted_ids))
