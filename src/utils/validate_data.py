"""
Data Validation Module for Telco Customer Churn Dataset

Validates data integrity, business logic constraints, and statistical properties
before model training.
"""

import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.
    
    Args:
        df: pandas DataFrame containing the telco customer data
        
    Returns:
        Tuple containing:
            - bool: True if all validations pass, False otherwise
            - List[str]: List of failed validation checks
    """
    print("Starting data validation!")
    
    failed_checks = []
    
    print("Validating schema and required columns!")
    
    # Check required columns exist.
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            failed_checks.append(f"Missing column: {col}")
    
    # Check for null values in customer ID.
    if "customerID" in df.columns and df["customerID"].isnull().any():
        failed_checks.append("customerID has null values")
    
    print("Validating business logic constraints!")
    
    # Gender validation.
    if "gender" in df.columns:
        valid_gender = df["gender"].isin(["Male", "Female"]).all()
        if not valid_gender:
            failed_checks.append("gender contains invalid values")
    
    # Yes/No fields validation.
    yes_no_fields = ["Partner", "Dependents", "PhoneService"]
    for field in yes_no_fields:
        if field in df.columns:
            valid = df[field].isin(["Yes", "No"]).all()
            if not valid:
                failed_checks.append(f"{field} contains invalid values")
    
    # Contract validation.
    if "Contract" in df.columns:
        valid_contract = df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all()
        if not valid_contract:
            failed_checks.append("Contract contains invalid values")
    
    # Internet service validation.
    if "InternetService" in df.columns:
        valid_internet = df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all()
        if not valid_internet:
            failed_checks.append("InternetService contains invalid values")
    
    print("Validating numeric ranges!")
    
    # Tenure validation.
    if "tenure" in df.columns:
        if df["tenure"].isnull().any():
            failed_checks.append("tenure has null values")
        elif (df["tenure"] < 0).any():
            failed_checks.append("tenure has negative values")
        elif (df["tenure"] > 120).any():
            failed_checks.append("tenure exceeds maximum (120 months)")
    
    # Monthly charges validation.
    if "MonthlyCharges" in df.columns:
        if df["MonthlyCharges"].isnull().any():
            failed_checks.append("MonthlyCharges has null values")
        elif (df["MonthlyCharges"] < 0).any():
            failed_checks.append("MonthlyCharges has negative values")
        elif (df["MonthlyCharges"] > 200).any():
            failed_checks.append("MonthlyCharges exceeds maximum (200)")
    
    # Total charges validation - Handle string values.
    if "TotalCharges" in df.columns:
        try:
            total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors='coerce')
            if (total_charges_numeric < 0).any():
                failed_checks.append("TotalCharges has negative values")
        except:
            failed_checks.append("TotalCharges contains non-numeric values")
    
    print("Validating data consistency!")
    
    # Total charges should be >= Monthly charges - Allow 5% exceptions.
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        try:
            total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors='coerce')
            comparison = total_charges_numeric >= df["MonthlyCharges"]
            if comparison.mean() < 0.95:
                failed_checks.append("TotalCharges < MonthlyCharges for more than 5% of records")
        except:
            pass
    
    # Print validation summary.
    total_checks = 15
    passed_checks = total_checks - len(failed_checks)
    
    is_valid = len(failed_checks) == 0
    
    if is_valid:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {len(failed_checks)}/{total_checks} checks failed")
        print(f"Failed checks: {failed_checks}")
    
    return is_valid, failed_checks

