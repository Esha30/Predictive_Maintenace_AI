# Predictive Maintenance AI

An end-to-end AI system that analyzes industrial machine sensor data, detects abnormal behavior, predicts the probability of machine failure, explains the prediction, and provides a professional web dashboard.

## Overview
This is a portfolio-grade AI Engineering project demonstrating a full machine learning lifecycle—from dataset ingestion and feature engineering to model training, explainability, API deployment, and a Next.js production dashboard.

## Problem Statement
Industrial machines fail unexpectedly, leading to costly downtime and emergency repairs. Traditional maintenance is either reactive (fix when broken) or time-based (fix every X months regardless of condition). Predictive maintenance solves this by analyzing real-time sensor data to predict *when* a machine is likely to fail, allowing for targeted, cost-effective maintenance.

## Solution
The Predictive Maintenance AI system uses:
1. **XGBoost Classifier** to predict the probability of failure based on sensor readings and engineered features.
2. **Isolation Forest** to detect anomalous sensor patterns that might indicate non-standard operation, even if not immediately indicative of failure.
3. **SHAP (SHapley Additive exPlanations)** to provide transparent, feature-level explanations of *why* a machine is flagged as at-risk.

## Features
- **Machine ML Pipeline**: Complete pipeline preventing data leakage, with proper stratification, scaling, and SMOTE for imbalanced classes.
- **REST API**: FastAPI backend providing endpoints for predictions, anomaly detection, machine management, and analytics.
- **Explainable AI**: Per-prediction SHAP explanations showing top contributing factors.
- **Modern Dashboard**: Responsive Next.js frontend with Tailwind CSS and Recharts for monitoring fleet health and prediction history.
- **Dockerized**: Fully containerized backend, frontend, and MongoDB database for easy deployment.

## Architecture
```
[ Sensor Data ] → [ FastAPI Backend ] ↔ [ ML Models (XGBoost, Isolation Forest) ]
                        ↓
                 [ MongoDB Database ]
                        ↓
                 [ Next.js Frontend ]
```

## Tech Stack
- **Machine Learning**: Python, Scikit-learn, XGBoost, SHAP, Pandas, Optuna, imbalanced-learn
- **Backend API**: FastAPI, Pydantic, Motor (Async MongoDB)
- **Database**: MongoDB
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts, Lucide React
- **Infrastructure**: Docker, Docker Compose

## Installation & Running Locally
The easiest way to run the entire stack is using Docker Compose.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Predictive-Maintenance-AI.git
cd Predictive-Maintenance-AI

# 2. Start the services
docker-compose up --build
```
- **Frontend Dashboard**: `http://localhost:3000`
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
- **MongoDB**: `localhost:27017`

*Note: If the ML models are not pre-trained, you must train them first by running `python -m ml.training.train` and `python -m ml.training.anomaly` in the backend environment, or map the models volume correctly.*
