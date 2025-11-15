import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_latest_model():
    """Load the most recent trained model"""
    model_dir = '../models'
    models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    
    if not models:
        raise FileNotFoundError("No trained models found")
    
    latest_model = sorted(models)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded model: {latest_model}")
    return model

def predict_fraud(transaction_data):
    """Predict if a transaction is fraudulent"""
    model = load_latest_model()
    
    # Make prediction
    prediction = model.predict(transaction_data)
    probability = model.predict_proba(transaction_data)[:, 1]
    
    result = {
        'is_fraud': bool(prediction[0]),
        'fraud_probability': float(probability[0]),
        'timestamp': datetime.now().isoformat()
    }
    
    return result

if __name__ == "__main__":
    # Example prediction
    np.random.seed(42)
    sample_transaction = pd.DataFrame({
        'Time': [12345],
        'Amount': [100.50],
        **{f'V{i}': [np.random.randn()] for i in range(1, 29)}
    })
    
    result = predict_fraud(sample_transaction)
    print(f"Prediction Result: {result}")