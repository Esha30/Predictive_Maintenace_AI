"""
Application settings — loaded from environment / .env file.

Risk level thresholds (configurable via environment variables):
  LOW      : failure_probability < RISK_LOW_THRESHOLD
  MEDIUM   : RISK_LOW_THRESHOLD  <= prob < RISK_MEDIUM_THRESHOLD
  HIGH     : RISK_MEDIUM_THRESHOLD <= prob < RISK_HIGH_THRESHOLD
  CRITICAL : prob >= RISK_HIGH_THRESHOLD
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── MongoDB ───────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "predictive_maintenance"

    # ── Models ────────────────────────────────────────────────
    model_path: str = "../ml/models"
    model_version: str = "1.0.0"

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ── Risk thresholds (0–1) ─────────────────────────────────
    risk_low_threshold: float = 0.30
    risk_medium_threshold: float = 0.50
    risk_high_threshold: float = 0.75

    # ── Anomaly ───────────────────────────────────────────────
    anomaly_contamination: float = 0.05

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def model_dir(self) -> Path:
        return Path(self.model_path)

    def classify_risk(self, probability: float) -> str:
        """Map failure probability to a risk level string."""
        if probability < self.risk_low_threshold:
            return "LOW"
        elif probability < self.risk_medium_threshold:
            return "MEDIUM"
        elif probability < self.risk_high_threshold:
            return "HIGH"
        else:
            return "CRITICAL"

    def recommended_action(self, risk_level: str) -> str:
        actions = {
            "LOW":      "Continue normal operations. Schedule routine inspection.",
            "MEDIUM":   "Monitor closely. Plan preventive maintenance within 2 weeks.",
            "HIGH":     "Schedule maintenance inspection within 48 hours.",
            "CRITICAL": "Immediate shutdown and emergency maintenance required.",
        }
        return actions.get(risk_level, "Review machine status.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
