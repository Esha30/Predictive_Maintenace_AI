# API Documentation

The Predictive Maintenance AI exposes a REST API powered by FastAPI.

## Endpoints

### 1. `GET /api/health`
Returns the health status of the API.
```json
{
  "status": "ok",
  "service": "Predictive Maintenance AI",
  "timestamp": "2026-09-22T10:00:00Z",
  "version": "1.0.0"
}
```

### 2. `POST /api/predict-and-analyze`
Runs the complete analysis pipeline (Failure Prediction + Anomaly Detection + SHAP Explanation).
**Request Body**:
```json
{
  "machine_id": "M-101",
  "machine_type": "M",
  "air_temp_k": 298.5,
  "process_temp_k": 308.6,
  "rotational_speed_rpm": 1551.0,
  "torque_nm": 42.8,
  "tool_wear_min": 197.0
}
```

**Response**:
```json
{
  "machine_id": "M-101",
  "failure_probability": 0.852,
  "risk_level": "CRITICAL",
  "prediction": "FAILURE_RISK",
  "recommended_action": "Immediate shutdown and emergency maintenance required.",
  "is_anomaly": true,
  "anomaly_score": 0.72,
  "anomaly_label": "Anomaly Detected",
  "shap_explanation": {
    "base_value": -1.2,
    "top_features": [
      { "feature": "torque_nm", "feature_value": 42.8, "shap_value": 0.34, "direction": "positive" }
    ],
    "all_contributions": []
  }
}
```

### 3. `GET /api/machines`
List all registered machines.

### 4. `GET /api/predictions`
List recent prediction history.

### 5. `GET /api/analytics/overview`
Get fleet-level key performance indicators.
