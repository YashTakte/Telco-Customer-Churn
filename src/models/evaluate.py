"""
Model Evaluation Module.
"""

import time
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)


def evaluate_model(model, X_test, y_test, threshold=0.35):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        threshold (float): Classification threshold (default: 0.35).
        
    Returns:
        tuple: (metrics_dict, predictions, probabilities)
    """
    print("Evaluating model performance!")
    
    # Generate predictions and track inference time.
    start_time = time.time()
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    pred_time = time.time() - start_time
    
    # Calculate all evaluation metrics.
    metrics = {
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "pred_time": pred_time,
        "samples_per_second": len(X_test) / pred_time
    }
    
    # Print performance summary.
    print(f"Model Performance:")
    print(f"   Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f} | ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Print detailed classification report.
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds, digits=3))
    
    # Print confusion matrix.
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    return metrics, preds, proba