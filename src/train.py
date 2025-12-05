import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

def load_data():
    """Load and prepare credit card fraud data"""

    print("Loading data...")

    df = pd.read_csv('/app/data/creditcard.csv')
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    print("Preprocessing data...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train the fraud detection model"""
    print("Training model...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment("fraud-detection-pipeline")
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        params = {
            'n_estimators': 300,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        
        print(f"Model trained successfully!")
        print(f"Metrics: {metrics}")
        
        return model, metrics

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, metrics = train_model(X_train, y_train, X_test, y_test)
