"""
Anomaly detection training using Isolation Forest.

The anomaly detector is trained ONLY on healthy machine observations
(machine_failure == 0) from the training set.

This is intentionally separate from the failure classifier because:
  - Failure prediction: supervised, answers "will this machine fail?"
  - Anomaly detection: unsupervised, answers "is this sensor reading unusual?"

An anomaly can exist without imminent failure (e.g., unusual vibration
that hasn't yet caused damage). Conversely, a failure can occur without
a previously detected anomaly (e.g., sudden random failure).

Usage:
  python -m ml.training.anomaly
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing.preprocessor import DataPreprocessor
from ml.preprocessing.feature_engineering import load_and_engineer

MODELS_DIR = ROOT / "ml" / "models"
CONTAMINATION = 0.05   # Expected fraction of anomalies (~5%)
RANDOM_STATE = 42


def train_anomaly_detector() -> None:
    """Train Isolation Forest on healthy training samples."""
    print("=" * 60)
    print("Anomaly Detection — Isolation Forest Training")
    print("=" * 60)

    # Load preprocessed training data
    train_csv = ROOT / "ml" / "data" / "processed" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            "Processed training data not found. Run ml/training/train.py first."
        )

    df_train = load_and_engineer(str(ROOT / "ml" / "data" / "raw" / "ai4i2020.csv"))

    # Filter to healthy machines only (no label leakage for unsupervised model)
    df_healthy = df_train[df_train["machine_failure"] == 0].reset_index(drop=True)
    print(f"Healthy training samples: {len(df_healthy)} / {len(df_train)}")

    # Load fitted scaler (already fitted on train in train.py)
    preprocessor = DataPreprocessor.load()
    X_healthy = preprocessor.transform(df_healthy)

    # Train Isolation Forest
    print(f"Training Isolation Forest (contamination={CONTAMINATION}) ...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_healthy)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "anomaly_detector.joblib"
    joblib.dump(iso_forest, model_path)
    print(f"Anomaly detector saved -> {model_path}")

    # Quick validation on training set (should have ~5% anomaly rate)
    all_csv = ROOT / "ml" / "data" / "raw" / "ai4i2020.csv"
    df_all = load_and_engineer(str(all_csv))

    # Use only training indices
    train_processed = pd.read_csv(train_csv)
    X_all_train = preprocessor.transform(df_all.iloc[: len(df_train)])

    preds = iso_forest.predict(X_all_train)
    anomaly_rate = (preds == -1).mean()
    print(f"Training anomaly rate: {anomaly_rate:.3%} (target ~{CONTAMINATION:.0%})")

    # Save anomaly detector metadata
    meta = {
        "model": "IsolationForest",
        "n_estimators": 200,
        "contamination": CONTAMINATION,
        "trained_on_healthy_only": True,
        "healthy_train_samples": int(len(df_healthy)),
        "anomaly_rate_train": float(anomaly_rate),
    }
    with open(MODELS_DIR / "anomaly_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Anomaly detector training complete.")


if __name__ == "__main__":
    train_anomaly_detector()
