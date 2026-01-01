"""
FastAPI + Gradio Application for Telco Churn Prediction.

Provides REST API and web UI for customer churn predictions.
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# Add parent directory to path for imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict


# Initialize FastAPI app.
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predict customer churn in telecom industry",
    version="1.0.0"
)


# Health check endpoint.
@app.get("/")
def root():
    """Health check for load balancers and monitoring."""
    return {"status": "ok"}


# Request schema for API.
class CustomerData(BaseModel):
    """Customer data schema for churn prediction."""
    
    # Demographics
    gender: str
    Partner: str
    Dependents: str
    
    # Phone services
    PhoneService: str
    MultipleLines: str
    
    # Internet services
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    
    # Account information
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    
    # Numeric features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# Prediction endpoint.
@app.post("/predict")
def api_predict(data: CustomerData):
    """
    Predict customer churn.
    
    Args:
        data: Customer information
        
    Returns:
        Dictionary with prediction result
    """
    try:
        prediction = predict(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


# Gradio interface function.
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """Process Gradio form inputs and return prediction."""
    
    # Construct customer data dictionary.
    customer_data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    
    # Get prediction.
    result = predict(customer_data)
    return str(result)


# Build Gradio web interface.
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        
        # Phone services
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        
        # Internet services
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        
        # Account information
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method"
        ),
        
        # Numeric features
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs="text",
    title="Telco Customer Churn Predictor",
    description="""
    Predict whether a customer is likely to churn based on their account details.
    
    Fill in the customer information below and click Submit to get a prediction. 
    The model uses XGBoost trained on historical telecom data.
    
    Note: Month-to-month contracts with fiber optic internet and electronic 
    check payments tend to have higher churn rates.
    """,
)

# Mount Gradio UI at /ui endpoint.
app = gr.mount_gradio_app(app, demo, path="/ui")