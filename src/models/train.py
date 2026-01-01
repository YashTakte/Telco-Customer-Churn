"""
Model Training Module.
"""

import time
from xgboost import XGBClassifier


def train_model(X_train, y_train, params):
    """
    Train XGBoost model with given parameters.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        params (dict): Model hyperparameters.
        
    Returns:
        tuple: (trained_model, training_time_in_seconds)
    """
    print("Training XGBoost model!")
    
    # Initialize model with provided parameters.
    model = XGBClassifier(**params)
    
    # Train model and track time.
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Model trained in {train_time:.2f} seconds")
    
    return model, train_time