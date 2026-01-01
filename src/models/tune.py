"""
Hyperparameter Tuning Module using Optuna.
"""

import os
import json
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def tune_model(X, y, n_trials=20, save_path=None):
    """
    Tune XGBoost model using Optuna optimization.
    
    Args:
        X: Training features.
        y: Training target.
        n_trials (int): Number of Optuna trials (default: 20).
        save_path (str): Path to save best parameters (optional).
        
    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    print(f"Starting Optuna hyperparameter tuning with {n_trials} trials!")
    
    def objective(trial):
        """Optuna objective function - Maximize recall score."""
        # Define hyperparameter search space.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        
        # Train model and evaluate with cross-validation.
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
    
    # Run Optuna optimization.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    
    # Print optimization results.
    print(f"Optuna tuning completed!")
    print(f"Best Recall: {study.best_value:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Save best parameters to file if path provided.
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Best parameters saved to: {save_path}")
    
    return best_params


def load_best_params(load_path):
    """
    Load best parameters from saved JSON file.
    
    Args:
        load_path (str): Path to JSON file containing parameters.
        
    Returns:
        dict: Loaded parameters, or None if file doesn't exist.
    """
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from: {load_path}")
        return params
    else:
        print(f"No saved parameters found at: {load_path}")
        return None