import mlflow
import pandas as pd
from datetime import datetime, timedelta
import os

def get_model_metrics():
    """Fetch metrics from MLflow"""
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    experiment = mlflow.get_experiment_by_name("fraud-detection-pipeline")
    
    if experiment is None:
        print("No experiment found")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        print("No runs found")
        return None
    
    # Get latest run metrics
    latest_run = runs.iloc[0]
    
    metrics = {
        'run_id': latest_run['run_id'],
        'accuracy': latest_run.get('metrics.accuracy', 0),
        'precision': latest_run.get('metrics.precision', 0),
        'recall': latest_run.get('metrics.recall', 0),
        'f1_score': latest_run.get('metrics.f1_score', 0),
        'roc_auc': latest_run.get('metrics.roc_auc', 0),
        'timestamp': latest_run['start_time']
    }
    
    return metrics

def check_model_drift():
    """Check if model needs retraining"""
    metrics = get_model_metrics()
    
    if metrics is None:
        return False, "No metrics available"
    
    # Define thresholds
    ACCURACY_THRESHOLD = 0.85
    F1_THRESHOLD = 0.80
    
    needs_retraining = (
        metrics['accuracy'] < ACCURACY_THRESHOLD or
        metrics['f1_score'] < F1_THRESHOLD
    )
    
    status = "NEEDS RETRAINING" if needs_retraining else "OK"
    
    print(f"Model Status: {status}")
    print(f"Current Metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
    
    return needs_retraining, metrics

if __name__ == "__main__":
    needs_retraining, metrics = check_model_drift()
    print(f"Retraining Required: {needs_retraining}")