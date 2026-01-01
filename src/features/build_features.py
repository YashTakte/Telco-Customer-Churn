import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:

    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.
    """

    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # === GENERIC BINARY MAPPING ===
    
    if len(vals) == 2:
        
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:

    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.
    """

    df = df.copy()

    # === Identify Feature Types ===

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"Found {len(obj_cols)} Categorical and {len(numeric_cols)} Numeric columns.")

    # === Split Categorical by Cardinality ===
    
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"Binary Features: {len(binary_cols)} | Multi-Category Features: {len(multi_cols)}")
    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-Category: {multi_cols}")

    # === Apply Binary Encoding ===
    
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"{c}: {original_dtype} -> Binary (0/1)")

    # === Convert Boolean Columns ===
    
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} Boolean columns to int: {bool_cols}")

    # === One-Hot Encoding for Multi-Category Features ===

    if multi_cols:
        print(f"Applying One-Hot Encoding to {len(multi_cols)} Multi-Category columns.")
        original_shape = df.shape
        
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} New features from {len(multi_cols)} Categorical columns.")

    # === Data Type Cleanup ===
    
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature Engineering Complete: {df.shape[1]} Final Features")
    return df

