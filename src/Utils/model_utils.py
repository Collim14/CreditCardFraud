from sklearn.metrics import (
    accuracy_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, precision_score, recall_score ,average_precision_score
)
import shap
import mlflow
import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate_metrics(y_true, y_probs):

    preds = (y_probs > 0.5).astype(int)
    
    metrics = {
        'PR_AUC': average_precision_score(y_true, y_probs),
        'F1': f1_score(y_true, preds),
        'Recall': recall_score(y_true, preds),
        'Precision': precision_score(y_true, preds),
        'Accuracy': accuracy_score(y_true, preds)
    }
    return metrics

def generate_shap_artifacts(model, X_sample, feature_names):
    """
    Generates SHAP plots only for the WINNING model to save compute time.
    """
    print("--- Generating SHAP Analysis for Champion Model ---")
    
    if hasattr(model, 'booster_') and model.booster_ is not None:
        booster = model.booster_
    else:
        return

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Champion Model)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close(fig)
    mlflow.log_artifact("shap_summary.png")

    fig_bar = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_bar.png")
    plt.close(fig_bar)
    mlflow.log_artifact("shap_bar.png")

    if os.path.exists("shap_summary.png"): os.remove("shap_summary.png")
    if os.path.exists("shap_bar.png"): os.remove("shap_bar.png")

class TimeSeriesValidator:
    def __init__(self, n_splits = 5):
        self.n_splits = n_splits

    def split(self, X, y):
        inds = np.arange(len(X))
        fold_size = len(X) // (self.n_splits + 1)
        for i in range(self.n_splits):
            train = fold_size * (i + 1)
            val = fold_size * (i + 2)
            train_idx = inds[:train]
            val_idx = inds[train:val]
            
            yield train_idx, val_idx