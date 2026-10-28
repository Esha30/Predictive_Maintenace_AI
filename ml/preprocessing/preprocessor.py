"""
Data preprocessing pipeline for the AI4I 2020 dataset.

Responsibilities:
  - Drop non-informative columns (UDI, product_id)
  - Handle missing values (median imputation for numerics)
  - Scale numeric features (StandardScaler)
  - Persist fitted scaler to disk (fitted on TRAIN only)

Design decisions:
  - Scaler is fit ONLY on training data, then applied to val/test
  - No SMOTE here — handled in train.py to avoid leakage
  - Returns numpy arrays + feature names list for SHAP compatibility
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from ml.preprocessing.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = Path(__file__).parent.parent / "models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# Columns to drop before training
DROP_COLUMNS = [
    "udi", "product_id", "type",
    "twf", "hdf", "pwf", "osf", "rnf",  # failure subtypes → encoded in target
]


class DataPreprocessor:
    """Fit-transform pipeline for sensor features."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False

    # ── Fit (training only) ───────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        X = self._select_features(df)
        self.scaler.fit(X)
        self._fitted = True
        return self

    # ── Transform (train / val / test / inference) ────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X = self._select_features(df)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    # ── Feature selection ─────────────────────────────────────
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = df[FEATURE_COLUMNS].copy()
        # Median imputation for any NaNs
        X = X.fillna(X.median(numeric_only=True))
        return X

    # ── Serialization ─────────────────────────────────────────
    def save(self, path: str | Path = SCALER_PATH) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path = SCALER_PATH) -> "DataPreprocessor":
        return joblib.load(path)

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_COLUMNS


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract binary target column."""
    return df[TARGET_COLUMN].values
