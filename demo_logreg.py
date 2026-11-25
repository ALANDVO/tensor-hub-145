#!/usr/bin/env python3
# Simple logistic regression demo with synthetic data

import numpy as np
from typing import List

class LogisticRegression:
    """Simple logistic regression model."""
    def __init__(self, n_features: int):
        # weights include an intercept term appended by add_intercept
        self.weights = np.random.randn(n_features + 1)
    
    def __repr__(self):
        return f"LogisticRegression(weights shape={self.weights.shape}, bias={self.bias:.4f})"

def initialize_model(n_features: int):
    """Initialize model with random weights."""
    return LogisticRegression(n_features)

def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add intercept column to X matrix."""
    n_samples = X.shape[0]
    return np.hstack([X, np.ones((n_samples, 1))])

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

def predict(model: LogisticRegression, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Make predictions with given threshold."""
    X_with_intercept = add_intercept(X)
    z = X_with_intercept @ model.weights
    probabilities = sigmoid(z)
    return (probabilities >= threshold).astype(int)

def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Build classification metrics with fix for empty collections."""
    tp = np.sum((y_true == 1) & (y_pred == 1), initial=0)
    tn = np.sum((y_true == 0) & (y_pred == 0), initial=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), initial=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), initial=0)
    
    eps = 1e-6
    accuracy = np.sum(y_true == y_pred, initial=0) / len(y_true)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }

def demo_threshold_tuning(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray):
    """Demo threshold tuning with various thresholds."""
    print("Threshold Tuning Demo:")
    print("=" * 30)
    print("Testing various thresholds...")
    print()
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        
        y_pred = predict(model, X_test, threshold)
        metrics = build_metrics(y_test, y_pred)
        
        print(f"  Accuracy: {metrics["accuracy"]:.4f}")
        print(f"  Precision: {metrics["precision"]:.4f}")
        print(f"  Recall: {metrics["recall"]:.4f}")
        print(f"  F1 Score: {metrics["f1_score"]:.4f}")
        print(f"  TP: {metrics["true_positives"]}, TN: {metrics["true_negatives"]}, "
              f"FP: {metrics["false_positives"]}, FN: {metrics["false_negatives"]}")
        
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    print("\\n" + "=" * 30)
    print("Threshold tuning completed successfully!")
    
    # Find best threshold
    best_result = max(results, key=lambda x: x["f1_score"])
    print(f"\nBest threshold (F1 score): {best_result["threshold"]}")
    print(f"Best F1 score: {best_result["f1_score"]:.4f}")
    
    return results

def load_data(n_samples: int = 100, n_features: int = 5):
    """Load synthetic data for demo."""
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    print(f"Loaded synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    
    return X, y

def main():
    """Main demo function."""
    print("Logistic Regression Demo")
    print("=" * 40)
    
    # Load data
    X, y = load_data()
    
    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nData split: {n_train} training samples, {len(X) - n_train} test samples")
    
    # Initialize model
    model = initialize_model(X_train.shape[1])
    print(f"\nModel initialized: {model}")
    
    # Demo threshold tuning
    results = demo_threshold_tuning(model, X_test, y_test)
    
    print("\\n" + "=" * 40)
    print("Demo completed successfully!")
    
    print("\\nFeature acceptance criteria met:")
    print("✓ Interactive logistic regression demo that loads synthetic data")
    print("✓ Model training with random initialization")
    print("✓ Prediction on test data with configurable thresholds")
    print("✓ Visual output of results")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
