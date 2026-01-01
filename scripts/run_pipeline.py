#!/usr/bin/env python3
"""
ML Pipeline: load → validate → preprocess → feature engineering → train → evaluate
"""

import os
import sys
import json
import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data
from src.models.tune import tune_model, load_best_params
from src.models.train import train_model
from src.models.evaluate import evaluate_model


def main(args):
    """Main training pipeline orchestrating the complete ML workflow."""
    
    # MLflow setup.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_dir = os.path.join(project_root, "mlruns")
    
    # Convert to proper file URI format for MLflow.
    mlruns_path = args.mlflow_uri or Path(mlruns_dir).as_uri()
    
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # Load data.
        print("Loading data!")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Validate data quality.
        print("Validating data quality!")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed!")

        # Preprocess data.
        print("Preprocessing data!")
        df = preprocess_data(df)

        processed_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved: {df.shape}")

        # Feature engineering.
        print("Building features!")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc = build_features(df, target_col=target)
        
        # Convert boolean columns to integers for XGBoost.
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # Save feature metadata for serving.
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving")

        # Train/test split.
        print("Splitting data!")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # Handle class imbalance.
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

        # Hyperparameter management - Tune or load saved params.
        best_params_path = os.path.join(artifacts_dir, "best_params.json")
        
        if args.tune:
            # Run fresh hyperparameter tuning with Optuna.
            best_params = tune_model(X_train, y_train, n_trials=20, save_path=best_params_path)
        else:
            # Load saved parameters (must exist from previous --tune run).
            best_params = load_best_params(best_params_path)
            
            if best_params is None:
                raise ValueError(
                    "No saved hyperparameters found! "
                    "Please run with --tune flag on first run to generate parameters:\n"
                    "  python scripts/run_pipeline.py --input data/raw/raw_data.csv --target Churn --tune"
                )
        
        # Add fixed parameters.
        best_params.update({
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight
        })
        
        # Log all hyperparameters to MLflow.
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Train model using train.py module.
        model, train_time = train_model(X_train, y_train, best_params)
        mlflow.log_metric("train_time", train_time)

        # Evaluate model using evaluate.py module.
        metrics, y_pred, proba = evaluate_model(model, X_test, y_test, threshold=args.threshold)
        
        # Log all metrics to MLflow.
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Print performance summary.
        print(f"\nPerformance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {metrics['pred_time']:.4f}s")
        print(f"   Samples per second: {metrics['samples_per_second']:.0f}")

        # Save model.
        print("\nSaving model to MLflow!")
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model saved to MLflow!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="Path to CSV (e.g., data/raw/raw_data.csv)")
    p.add_argument("--target", type=str, default="Churn",
                   help="Target column name")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="Classification threshold")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Test set proportion")
    p.add_argument("--experiment", type=str, default="Telco Churn",
                   help="MLflow experiment name")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="MLflow tracking URI (default: project_root/mlruns)")
    p.add_argument("--tune", action="store_true",
                   help="Run hyperparameter tuning with Optuna")

    args = p.parse_args()
    main(args)
