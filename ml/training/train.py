"""
Model training script for Predictive Maintenance AI.

Trains three models in sequence:
  1. Logistic Regression  (baseline)
  2. Random Forest        (baseline ensemble)
  3. XGBoost              (advanced, with Optuna HPO)

Data leakage prevention:
  - Train/Val/Test split is stratified and performed BEFORE any fitting.
  - The preprocessor (scaler) is fit ONLY on X_train.
  - Optuna uses X_val for early stopping — X_test is untouched until evaluation.
  - SMOTE is applied ONLY to the training set after splitting.

Usage:
  python -m ml.training.train
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path setup ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.data.download_dataset import download_dataset
from ml.preprocessing.feature_engineering import load_and_engineer
from ml.preprocessing.preprocessor import DataPreprocessor, get_labels

MODELS_DIR = ROOT / "ml" / "models"
PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
SPLIT_INFO_PATH = MODELS_DIR / "split_info.json"

# ── Constants ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15   # of remaining after test split
N_TRIALS = 50     # Optuna trials for XGBoost HPO


# ─────────────────────────────────────────────────────────
# 1.  Load + engineer features
# ─────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    csv_path = ROOT / "ml" / "data" / "raw" / "ai4i2020.csv"
    if not csv_path.exists():
        csv_path = download_dataset()
    return load_and_engineer(str(csv_path))


# ─────────────────────────────────────────────────────────
# 2.  Split (stratified, leak-free)
# ─────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    y = get_labels(df)

    # First: carve out test set
    idx = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Then: split remaining into train / val
    y_trainval = y[idx_trainval]
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=y_trainval,
        random_state=RANDOM_STATE
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    print(
        f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )
    print(
        f"Class distribution — train: {df_train['machine_failure'].value_counts().to_dict()}, "
        f"val: {df_val['machine_failure'].value_counts().to_dict()}, "
        f"test: {df_test['machine_failure'].value_counts().to_dict()}"
    )
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────
# 3.  Preprocess (scaler fit on train only)
# ─────────────────────────────────────────────────────────
def preprocess(df_train, df_val, df_test):
    preprocessor = DataPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_val   = preprocessor.transform(df_val)
    X_test  = preprocessor.transform(df_test)

    y_train = get_labels(df_train)
    y_val   = get_labels(df_val)
    y_test  = get_labels(df_test)

    # SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — X_train: {X_train_res.shape}, class balance: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    preprocessor.save()
    return preprocessor, X_train_res, y_train_res, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────
# 4.  Baseline models
# ─────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("\nTraining Logistic Regression ...")
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────
# 5.  XGBoost with Optuna HPO
# ─────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    print("Running Optuna HPO for XGBoost ...")

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        from sklearn.metrics import roc_auc_score
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })

    print(f"Best ROC-AUC (val): {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=False)
    return final_model


# ─────────────────────────────────────────────────────────
# 6.  Save
# ─────────────────────────────────────────────────────────
def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved -> {path}")


def save_split_info(df_train, df_val, df_test) -> None:
    info = {
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "total": len(df_train) + len(df_val) + len(df_test),
        "train_positive_rate": float(df_train["machine_failure"].mean()),
        "val_positive_rate": float(df_val["machine_failure"].mean()),
        "test_positive_rate": float(df_test["machine_failure"].mean()),
        "random_state": RANDOM_STATE,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)


def save_processed_data(df_train, df_val, df_test) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "test.csv", index=False)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Predictive Maintenance AI — Training Pipeline")
    print("=" * 60)

    df = load_data()
    print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns")

    df_train, df_val, df_test = split_data(df)
    save_processed_data(df_train, df_val, df_test)
    save_split_info(df_train, df_val, df_test)

    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test = preprocess(
        df_train, df_val, df_test
    )

    # Train all models
    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train, X_val, y_val)

    # Save
    save_model(lr,  "logistic_regression")
    save_model(rf,  "random_forest")
    save_model(xgb, "xgboost")

    # Save test arrays for evaluation script
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)
    np.save(MODELS_DIR / "X_val.npy",  X_val)
    np.save(MODELS_DIR / "y_val.npy",  y_val)

    print("\n✅ Training complete. Run `python -m ml.evaluation.evaluate` next.")


if __name__ == "__main__":
    main()
