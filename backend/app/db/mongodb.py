"""
MongoDB async client setup using Motor.

Usage:
  from app.db.mongodb import get_db
  db = get_db()
  collection = db["machines"]
"""

from __future__ import annotations

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        raise RuntimeError("MongoDB client not initialised. Call connect_db() first.")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return get_client()[settings.database_name]


async def connect_db() -> None:
    global _client
    settings = get_settings()
    logger.info("Connecting to MongoDB at %s …", settings.mongodb_uri)
    _client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )
    # Verify connection
    await _client.admin.command("ping")
    logger.info("MongoDB connected — database: %s", settings.database_name)


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed.")
