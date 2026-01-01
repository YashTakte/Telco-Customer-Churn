"""
Inference Pipeline for Telco Churn Prediction.

Loads trained model and applies same transformations used during training.
"""

import os
import pandas as pd
import mlflow


# Model loading configuration.
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}.")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}.")
    try:
        import glob
        
        # First try: src/serving/model directory (for local testing with copied model)
        serving_model_path = os.path.join("src", "serving", "model", "*", "artifacts", "model")
        serving_models = glob.glob(serving_model_path)
        
        if serving_models:
            latest_model = max(serving_models, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Loaded model from serving directory: {latest_model}.")
        else:
            # Fallback: mlruns directory
            local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
            if local_model_paths:
                latest_model = max(local_model_paths, key=os.path.getmtime)
                model = mlflow.pyfunc.load_model(latest_model)
                MODEL_DIR = latest_model
                print(f"Fallback: Loaded model from mlruns: {latest_model}.")
            else:
                raise Exception("No model found in /app/model, src/serving/model, or mlruns.")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}.")


# Load feature schema.
try:
    # First try: Load from MLflow model directory (Docker/production)
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if os.path.exists(feature_file):
        with open(feature_file) as f:
            FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(FEATURE_COLS)} feature columns from MLflow model.")
    else:
        # Fallback: Load from artifacts directory (local development)
        feature_file_fallback = os.path.join("artifacts", "feature_columns.json")
        if os.path.exists(feature_file_fallback):
            import json
            with open(feature_file_fallback) as f:
                FEATURE_COLS = json.load(f)
            print(f"Loaded {len(FEATURE_COLS)} feature columns from artifacts/feature_columns.json.")
        else:
            raise Exception(f"feature_columns not found in {feature_file} or {feature_file_fallback}")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}.")


# Feature transformation constants.
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply same feature transformations used during training.
    
    Steps:
    1. Convert numeric columns to proper types
    2. Apply binary encoding (Yes/No, Male/Female)
    3. One-hot encode remaining categorical features
    4. Convert booleans to integers
    5. Align with training feature schema
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns.
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)
    
    # Apply binary encoding.
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )
    
    # One-hot encode categorical features.
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # Convert booleans to integers.
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # Align with training features.
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df


def predict(input_dict: dict) -> str:
    """
    Predict customer churn from input data.
    
    Args:
        input_dict: Customer data dictionary
        
    Returns:
        "Likely to Churn" or "Not Likely to Churn"
    """
    # Convert input to DataFrame.
    df = pd.DataFrame([input_dict])
    
    # Apply feature transformations.
    df_enc = _serve_transform(df)
    
    # Generate prediction.
    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
    
    except Exception as e:
        raise Exception(f"Model Prediction Failed: {e}")
    
    # Convert prediction to user-friendly output.
    if result == 1:
        return "Likely to Churn"
    else:
        return "Not Likely to Churn"