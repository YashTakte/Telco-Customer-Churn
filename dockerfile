FROM python:3.11-slim

# Set working directory inside the container.
WORKDIR /app

COPY requirements.txt .

# Install Python dependencies.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the image.
COPY . .

# Explicitly copy model 
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path.
COPY src/serving/model/162f25c8a2af49f6b0b33a7af7b719c2/artifacts/model /app/model
COPY src/serving/model/162f25c8a2af49f6b0b33a7af7b719c2/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/162f25c8a2af49f6b0b33a7af7b719c2/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# Set environment variables.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Expose FastAPI port.
EXPOSE 8000

# Run the FastAPI app using uvicorn.
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
