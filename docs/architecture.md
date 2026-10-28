# Architecture Documentation

## System Components

### 1. ML Pipeline (`ml/`)
The ML pipeline is responsible for data ingestion, preprocessing, training, and evaluation.
- **Leakage Prevention**: The pipeline strictly separates train/validation/test sets before any transformations. The scaler (DataPreprocessor) is fitted ONLY on the training data.
- **Class Imbalance**: The AI4I 2020 dataset has a high class imbalance (few failures). SMOTE is applied *only* to the training set to prevent synthetic data from bleeding into validation/test metrics.
- **Hyperparameter Tuning**: Optuna is used to optimize the XGBoost model using a validation set.

### 2. FastAPI Backend (`backend/`)
The backend serves the ML models and provides a REST API.
- **Model Registry**: Models (`.joblib` files) are loaded once at startup into a singleton `ModelRegistry` to ensure low-latency inference.
- **Predictor Service**: Bridges the raw API JSON payload with the ML feature engineering pipeline.
- **Database Layer**: Uses Motor (async driver) for MongoDB.

### 3. Next.js Frontend (`frontend/`)
The frontend is a modern React application using the Next.js App Router.
- **Server Components**: Used where possible for data fetching.
- **Recharts**: For rendering time-series sensor data and SHAP contribution charts.
- **Tailwind CSS**: For utility-first styling.

### 4. MongoDB
Stores:
- `machines`: Machine metadata and current status.
- `predictions`: History of all predictions, anomaly scores, and risk levels.
