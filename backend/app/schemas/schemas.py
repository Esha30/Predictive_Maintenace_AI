"""
Pydantic schemas for request and response validation.

SensorInput       — what the API receives for predictions
PredictionResponse — what the API returns for predictions
MachineSchema     — machine document
SensorReadingSchema — sensor snapshot stored in DB
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────
# Sensor Input
# ──────────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    """Raw sensor readings for a single prediction request."""

    machine_id: Optional[str] = Field(None, description="Machine identifier (optional)")
    machine_type: Literal["L", "M", "H"] = Field(
        "M", description="Machine quality type: L=Low, M=Medium, H=High"
    )
    air_temp_k: float = Field(
        ..., ge=290.0, le=320.0,
        description="Air temperature in Kelvin (typical: 295–305 K)"
    )
    process_temp_k: float = Field(
        ..., ge=300.0, le=340.0,
        description="Process temperature in Kelvin (typical: 305–315 K)"
    )
    rotational_speed_rpm: float = Field(
        ..., ge=1000.0, le=3000.0,
        description="Rotational speed in RPM (typical: 1200–2800)"
    )
    torque_nm: float = Field(
        ..., ge=3.0, le=80.0,
        description="Torque in Newton-metres (typical: 10–70 Nm)"
    )
    tool_wear_min: float = Field(
        ..., ge=0.0, le=300.0,
        description="Tool wear time in minutes (typical: 0–250)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_id": "M-104",
                "machine_type": "M",
                "air_temp_k": 298.5,
                "process_temp_k": 308.6,
                "rotational_speed_rpm": 1551.0,
                "torque_nm": 42.8,
                "tool_wear_min": 197.0,
            }
        }
    }


# ──────────────────────────────────────────────────────────────
# SHAP feature contribution
# ──────────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: Literal["positive", "negative"]


class SHAPExplanation(BaseModel):
    base_value: float
    top_features: List[FeatureContribution]
    all_contributions: List[FeatureContribution]


# ──────────────────────────────────────────────────────────────
# Prediction Response
# ──────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    machine_id: Optional[str] = None
    failure_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    prediction: Literal["NORMAL", "FAILURE_RISK"]
    recommended_action: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"


class AnomalyResponse(BaseModel):
    machine_id: Optional[str] = None
    is_anomaly: bool
    anomaly_score: float
    anomaly_label: Literal["Normal", "Anomaly Detected"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FullAnalysisResponse(PredictionResponse):
    is_anomaly: bool
    anomaly_score: float
    anomaly_label: Literal["Normal", "Anomaly Detected"]
    shap_explanation: SHAPExplanation


# ──────────────────────────────────────────────────────────────
# Machine schemas
# ──────────────────────────────────────────────────────────────

class MachineCreate(BaseModel):
    machine_id: str
    machine_name: str
    machine_type: str
    installation_date: Optional[str] = None
    location: Optional[str] = None
    status: str = "healthy"
    operating_hours: int = 0


class MachineSchema(MachineCreate):
    last_maintenance: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────────────────────
# Analytics schemas
# ──────────────────────────────────────────────────────────────

class OverviewStats(BaseModel):
    total_machines: int
    healthy: int
    warning: int
    critical: int
    total_predictions: int
    anomalies_detected: int
    avg_failure_probability: float


class ModelInfo(BaseModel):
    primary_model: str
    model_version: str
    feature_count: int
    feature_names: List[str]
    metrics: dict[str, Any]
    models_available: List[str]
    is_ready: bool
