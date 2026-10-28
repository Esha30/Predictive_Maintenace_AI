"""
Feature engineering for the AI4I 2020 Predictive Maintenance Dataset.

Creates meaningful derived features from raw sensor readings.
All transformations are deterministic and leak-free — they depend only
on values within the same row (no future data, no test-set statistics).
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────
# Column name mapping  (raw UCI → clean internal names)
# ──────────────────────────────────────────────────────────
COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",   # Tool Wear Failure
    "HDF": "hdf",   # Heat Dissipation Failure
    "PWF": "pwf",   # Power Failure
    "OSF": "osf",   # Overstrain Failure
    "RNF": "rnf",   # Random Failure
}

# Features used for model training (after engineering)
FEATURE_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    # engineered
    "temp_delta_k",
    "power_kw",
    "tool_wear_torque",
    "type_encoded",
    "temp_ratio",
    "speed_torque_ratio",
]

TARGET_COLUMN = "machine_failure"

# Type encoding: L=0, M=1, H=2  (ordinal — H implies higher spec)
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw UCI columns to clean internal names."""
    return df.rename(columns=COLUMN_MAP)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw sensor readings.

    New features
    ─────────────
    temp_delta_k        Process temp − Air temp  (heat dissipation indicator)
    power_kw            Rotational speed × Torque / 9549  (mechanical power)
    tool_wear_torque    tool_wear × torque  (interaction: stress on worn tool)
    type_encoded        Ordinal encoding of machine Type (L=0, M=1, H=2)
    temp_ratio          Process temp / Air temp
    speed_torque_ratio  Rotational speed / (Torque + 1e-6)
    """
    df = df.copy()

    # Ordinal encoding for machine Type
    df["type_encoded"] = df["type"].map(TYPE_ENCODING).fillna(0).astype(int)

    # Temperature differential
    df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]

    # Mechanical power (W → kW)
    df["power_kw"] = (df["rotational_speed_rpm"] * df["torque_nm"]) / 9549.0

    # Interaction term: stressed worn tool
    df["tool_wear_torque"] = df["tool_wear_min"] * df["torque_nm"]

    # Temperature ratio
    df["temp_ratio"] = df["process_temp_k"] / (df["air_temp_k"] + 1e-6)

    # Speed-torque ratio (inverse of load)
    df["speed_torque_ratio"] = df["rotational_speed_rpm"] / (df["torque_nm"] + 1e-6)

    return df


def load_and_engineer(raw_csv_path: str) -> pd.DataFrame:
    """
    Full pipeline: load CSV → rename columns → engineer features.
    Returns a clean DataFrame ready for preprocessing.
    """
    df = pd.read_csv(raw_csv_path)
    df = rename_columns(df)
    df = engineer_features(df)
    return df
