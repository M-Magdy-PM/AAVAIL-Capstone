# AAVAIL Time-Series Revenue Forecasting Pipeline and REST API

## Project Overview
This repository contains the implementation for the AAVAIL Capstone Project, focusing on the productionization of a time-series revenue forecasting system. The project integrates automated data ingestion, advanced feature engineering, optimized supervised learning models via Scikit-Learn, a structured logging mechanism, a production-grade Flask REST API, and containerization assets for deployment.

## Repository Architecture

```text
ai-workflow-capstone/
│
├── cs-train/              # Raw monthly JSON invoice data and processed time-series CSVs
│   └── ts-data/           # Cached daily time-series aggregates per country
├── models/                # Saved serialized model artifacts (.joblib)
├── logs/                  # CSV logs tracking training sessions and API predictions
│
├── cslib.py               # Data ingestion, cleaning, and time-series feature engineering
├── model.py               # Model training (GridSearchCV), evaluation, loading, and prediction
├── logger.py              # Centralized logging management for training and predictions
├── app.py                 # Production-grade Flask REST API server
├── unittests.py           # Comprehensive automated unit test suite
├── Dockerfile             # Container configuration for deployment
└── requirements.txt       # Project dependency manifest
```

## Core Components

- **`cslib.py` (Data and Feature Engineering)**: Provides automated path resolution and aggregates raw transaction records into daily time-series metrics. It extracts lag features and rolling window statistics required for regression modeling.
- **`model.py` (Machine Learning Pipeline)**: Implements Scikit-Learn pipelines combining data scaling and Random Forest Regressors, optimized via grid search. Handles evaluation metrics (RMSE), model serialization, and batch inference.
- **`logger.py` (Activity Tracking)**: Records execution telemetry, performance evaluations, and prediction audit trails into structured CSV log files.
- **`app.py` (Flask REST API)**: Exposes endpoints for health checks, live revenue predictions, model retraining, and log inspection.
- **`unittests.py` (Automated Testing)**: Verifies data ingestion integrity, feature engineering logic, model inference accuracy, and all HTTP endpoints.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/M-Magdy-PM/AAVAIL-Capstone.git
cd AAVAIL-Capstone
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## API Endpoints Reference

| Endpoint | Method | Description | Example Query Parameters / Payload |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | API health check and available models list | None |
| `/predict` | `GET` / `POST` | Generate revenue prediction for a target date | `?country=all&year=2018&month=01&day=05&test=true` |
| `/train` | `POST` / `GET` | Trigger retraining of models on available data | JSON: `{"test": true}` |
| `/logs/<log_type>` | `GET` | Retrieve training or prediction logs (`train` or `predict`) | `?test=true` |

## Running Unit Tests
Execute the automated test suite to validate all backend and API operations:
```bash
python unittests.py
```

## Docker Deployment
Build and run the application container locally:
```bash
docker build -t aavail-revenue-api .
docker run -p 5000:5000 aavail-revenue-api
```
