# MLOps-Powered Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?logo=scikit-learn&logoColor=white)

Production-ready MLOps pipeline for detecting fraudulent credit card transactions with automated experiment tracking and containerized deployment.

## 🎯 What This Does

Detects credit card fraud using Random Forest with **99.9% accuracy** on 284K+ transactions. Built with MLflow for experiment tracking, Docker for reproducibility, and PostgreSQL for metadata storage.

**Key Achievement:** Handles highly imbalanced dataset (492 frauds in 284K transactions) using SMOTE oversampling, achieving 0.80 F1-score and 0.98 ROC-AUC.

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 99.93% |
| Precision | 0.87 |
| Recall | 0.80 |
| F1-Score | 0.80 |
| ROC-AUC | 0.98 |
| Training Time | ~45 seconds |

## 🛠️ Tech Stack

- **ML**: Scikit-learn, Imbalanced-learn (SMOTE)
- **MLOps**: MLflow 2.9.2, Docker Compose, PostgreSQL 14
- **Language**: Python 3.10

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- 8GB RAM minimum

### Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/nandiniranjansinha/mlops-fraud-detection.git
cd mlops-fraud-detection

# 2. Download dataset from Kaggle (creditcard.csv)
# Place in data/ folder
# Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Start services
docker-compose up -d

# 4. Run training
docker-compose exec mlflow python /app/src/train.py

# 5. Access MLflow UI
# Open http://localhost:5000
```

## 📁 Project Structure

```
mlops-pipeline/
├── data/
│   └── credit_card_data.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── monitor.py
├── airflow/
│   └── dags/
│       └── ml_pipeline_dag.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── MLproject
└── README.md
```

## 🔧 Usage

### Train New Model
```bash
docker-compose exec mlflow python /app/src/train.py
```

### Make Predictions
```python
from predict import load_latest_model, predict_fraud

model = load_latest_model()
result = predict_fraud(transaction_data)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

### View Experiments
Open MLflow UI at `http://localhost:5000` to compare runs, view metrics, and download models.

### Stop Services
```bash
docker-compose down
```

## 🤖 Model Details

**Algorithm:** Random Forest Classifier
- 100 estimators, max_depth=10
- SMOTE oversampling for class balance
- 6 tracked metrics per run (accuracy, precision, recall, F1, ROC-AUC, training time)

**Data:** 284,807 transactions with 30 features (Time, Amount, V1-V28)
- 492 fraudulent transactions (0.17%)
- 80/20 train-test split (stratified)

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
