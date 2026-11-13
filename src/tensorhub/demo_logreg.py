#!/usr/bin/env python3
import numpy as np
def build_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_pred)) == 1:
        tp = np.sum((y_true == 1).astype(int) * (y_pred == 1).astype(int))
        fp = np.sum((y_true == 0).astype(int) * (y_pred == 1).astype(int))
        tn = np.sum((y_true == 0).astype(int) * (y_pred == 0).astype(int))
        fn = np.sum((y_true == 1).astype(int) * (y_pred == 0).astype(int))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        return accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)
if __name__ == "__main__":
    yt = np.array([1,0,1])
    yp = np.array([0,0,0])
    print(build_metrics(yt, yp))
