# Telco Customer Churn Prediction

A machine learning system that predicts telecom customer churn with 86% recall and 84% ROC-AUC on 7,000+ customers, deployed as a REST API with automated CI/CD.

**Base URL:** https://telco-customer-churn-prediction-f2tf.onrender.com

**Web Interface:** https://telco-customer-churn-prediction-f2tf.onrender.com/ui

---

## Model Performance

| Metric | Value |
|--------|-------|
| Recall | 86% |
| ROC-AUC | 84% |
| Dataset | 7,043 customers |
| Throughput | 466K predictions/second |

High recall prioritized to minimize business costs of missed churners.

---

## Quick Start

### Use Live Application

1. First, visit the base URL to wake up the service: https://telco-customer-churn-prediction-f2tf.onrender.com
2. Then access the web interface: https://telco-customer-churn-prediction-f2tf.onrender.com/ui

### Run Locally

```bash
# Clone and setup
git clone https://github.com/YashTakte/Telco-Customer-Churn.git
cd Telco-Customer-Churn
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install and run
pip install -r requirements.txt
uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Access at http://localhost:8000/ui
```

### Docker

```bash
docker pull yashtakte/telco-churn-api:latest
docker run -p 8000:8000 yashtakte/telco-churn-api:latest
```

---

## API Usage

**Health Check:**
```bash
curl https://telco-customer-churn-prediction-f2tf.onrender.com/
```

**Prediction:**
```bash
curl -X POST "https://telco-customer-churn-prediction-f2tf.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1,
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0
  }'
```

---

## Project Structure

```
Telco-Customer-Churn/
├── .github/workflows/     # CI/CD pipeline
├── notebooks/
│   └── EDA.ipynb         # Exploratory data analysis
├── scripts/
│   ├── prepare_processed_data.py
│   ├── run_pipeline.py
│   └── test_fastapi.py
├── src/
│   ├── app/              # FastAPI application
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # Training, evaluation, tuning
│   ├── serving/
│   │   ├── __pycache__/
│   │   ├── model/        # Trained model artifacts
│   │   └── inference.py  # Model inference
│   └── utils/            # Utility functions
├── Dockerfile           
└── requirements.txt      
```

---

## Technology Stack

**Machine Learning:**
- XGBoost - Model algorithm
- Scikit-learn - Preprocessing
- Optuna - Hyperparameter tuning
- MLflow - Experiment tracking

**Web Framework:**
- FastAPI - REST API
- Gradio - Web interface
- Pydantic - Data validation

**DevOps:**
- Docker - Containerization
- GitHub Actions - CI/CD
- Render - Cloud hosting

---

## Training

```bash
# With hyperparameter tuning
python scripts/run_pipeline.py --input data/raw/raw_data.csv --target Churn --tune

# Using saved parameters
python scripts/run_pipeline.py --input data/raw/raw_data.csv --target Churn

# View results
mlflow ui
```

---

## Deployment Pipeline

```
GitHub → GitHub Actions → Docker Hub → Render → Production
```

Automated deployment on push to main branch.

---

## Links

- Base URL: https://telco-customer-churn-prediction-f2tf.onrender.com
- Web App: https://telco-customer-churn-prediction-f2tf.onrender.com/ui
- API Docs: https://telco-customer-churn-prediction-f2tf.onrender.com/docs
- GitHub: https://github.com/YashTakte/Telco-Customer-Churn
- Docker Hub: https://hub.docker.com/r/yashtakte/telco-churn-api

---
