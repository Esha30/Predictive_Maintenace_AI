"""Tests for the ML pipeline (no DB, no API)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestFeatureEngineering:
    """Tests for ml.preprocessing.feature_engineering."""

    def test_rename_columns(self):
        from ml.preprocessing.feature_engineering import rename_columns, COLUMN_MAP
        raw_cols = list(COLUMN_MAP.keys())
        df = pd.DataFrame([{c: 0 for c in raw_cols}])
        renamed = rename_columns(df)
        assert "air_temp_k" in renamed.columns
        assert "Air temperature [K]" not in renamed.columns

    def test_engineer_features_creates_expected_cols(self):
        from ml.preprocessing.feature_engineering import rename_columns, engineer_features, FEATURE_COLUMNS
        raw = {
            "UDI": 1, "Product ID": "M001", "Type": "M",
            "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8,
            "Tool wear [min]": 197, "Machine failure": 0,
            "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        }
        df = pd.DataFrame([raw])
        df = rename_columns(df)
        df = engineer_features(df)
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_power_calculation(self):
        from ml.preprocessing.feature_engineering import rename_columns, engineer_features
        raw = {
            "UDI": 1, "Product ID": "M001", "Type": "M",
            "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1000.0, "Torque [Nm]": 10.0,
            "Tool wear [min]": 0, "Machine failure": 0,
            "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        }
        df = rename_columns(pd.DataFrame([raw]))
        df = engineer_features(df)
        expected_power = (1000.0 * 10.0) / 9549.0
        assert abs(df["power_kw"].iloc[0] - expected_power) < 1e-6

    def test_type_encoding(self):
        from ml.preprocessing.feature_engineering import rename_columns, engineer_features
        for type_val, expected in [("L", 0), ("M", 1), ("H", 2)]:
            raw = {
                "UDI": 1, "Product ID": "A1", "Type": type_val,
                "Air temperature [K]": 298.0, "Process temperature [K]": 308.0,
                "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0,
                "Tool wear [min]": 100, "Machine failure": 0,
                "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
            }
            df = rename_columns(pd.DataFrame([raw]))
            df = engineer_features(df)
            assert df["type_encoded"].iloc[0] == expected


class TestPreprocessor:
    """Tests for ml.preprocessing.preprocessor.DataPreprocessor."""

    def _make_df(self, n=5):
        from ml.preprocessing.feature_engineering import rename_columns, engineer_features
        rows = []
        for i in range(n):
            rows.append({
                "UDI": i, "Product ID": f"M{i:03d}", "Type": "M",
                "Air temperature [K]": 298.0 + i * 0.1,
                "Process temperature [K]": 308.0 + i * 0.1,
                "Rotational speed [rpm]": 1500.0 + i * 10,
                "Torque [Nm]": 40.0 + i,
                "Tool wear [min]": 100 + i * 5,
                "Machine failure": 0,
                "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
            })
        df = pd.DataFrame(rows)
        df = rename_columns(df)
        df = engineer_features(df)
        return df

    def test_fit_transform_shape(self):
        from ml.preprocessing.preprocessor import DataPreprocessor, FEATURE_COLUMNS
        df = self._make_df(10)
        prep = DataPreprocessor()
        X = prep.fit_transform(df)
        assert X.shape == (10, len(FEATURE_COLUMNS))

    def test_transform_without_fit_raises(self):
        from ml.preprocessing.preprocessor import DataPreprocessor
        df = self._make_df(5)
        prep = DataPreprocessor()
        with pytest.raises(RuntimeError):
            prep.transform(df)

    def test_feature_names_property(self):
        from ml.preprocessing.preprocessor import DataPreprocessor, FEATURE_COLUMNS
        prep = DataPreprocessor()
        assert prep.feature_names == FEATURE_COLUMNS

    def test_missing_column_raises(self):
        from ml.preprocessing.preprocessor import DataPreprocessor
        df = self._make_df(3)
        prep = DataPreprocessor()
        prep.fit(df)
        broken_df = df.drop(columns=["air_temp_k"])
        with pytest.raises(ValueError):
            prep.transform(broken_df)
