# Model & Data Documentation

## Dataset
**Source**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
**Records**: 10,000
**Features**: 14 (6 raw sensor features + metadata + failure flags)
**Target**: Machine failure (binary)

## Feature Engineering
The pipeline derives new features from the raw sensors to capture mechanical and thermodynamic interactions:
- `temp_delta_k`: Process temperature − Air temperature (heat dissipation indicator)
- `power_kw`: Rotational speed × Torque / 9549 (mechanical power)
- `tool_wear_torque`: Tool wear × torque (interaction term for stress on a worn tool)
- `temp_ratio`: Process temp / Air temp
- `speed_torque_ratio`: Rotational speed / Torque

## Model Comparison
Three classifiers were evaluated on a strict hold-out test set:
1. **Logistic Regression** (Baseline)
2. **Random Forest** (Baseline Ensemble)
3. **XGBoost** (Advanced, optimized via Optuna)

XGBoost was selected as the primary model due to its superior ROC-AUC and PR-AUC scores, particularly its ability to balance precision and recall on the minority (failure) class.

## Anomaly Detection
An Isolation Forest is trained exclusively on *healthy* data. This provides an unsupervised anomaly score that can detect unusual operating conditions even if the XGBoost classifier does not predict an imminent failure.
