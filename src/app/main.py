"""
FastAPI + Gradio Application for Telco Churn Prediction.

Provides REST API and web UI for customer churn predictions.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict


# Initialize FastAPI application.
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)


# Health check endpoint (required for AWS Load Balancer).
@app.get("/")
def root():
    """Health check for monitoring and load balancers."""
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


# Prediction API endpoint.
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Predict customer churn.
    
    Returns:
        {"prediction": "Likely to Churn"} or {"prediction": "Not Likely to Churn"}
    """
    try:
        result = predict(data.dict())
        return {"prediction": result}
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
    data = {
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
    result = predict(data)
    return str(result)


# Build Gradio web interface.
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        
        # Phone services
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        
        # Internet services
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        
        # Account information
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        
        # Numeric features
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Output", lines=2),
    title="Telco Customer Churn Predictor",
    description="""
    Predict whether a customer is likely to churn based on their account details.
    
    Fill in the customer information below and click Submit to get a prediction. 
    The model uses XGBoost trained on historical telecom data.
    
    Note: Month-to-month contracts with fiber optic internet and electronic 
    check payments tend to have higher churn rates.
    """,
    examples=[
        # High churn risk example
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", 
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 
         1, 85.0, 85.0],
        # Low churn risk example
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
    ],
    theme=gr.themes.Soft()
)

# Mount Gradio UI at /ui endpoint.
app = gr.mount_gradio_app(app, demo, path="/ui")
