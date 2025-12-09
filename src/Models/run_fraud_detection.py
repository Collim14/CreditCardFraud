

import pandas as pd
import numpy as np
import os
import time
from random import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from mlflow.models.signature import infer_signature

import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, auc, 
    confusion_matrix, f1_score, precision_score, recall_score ,average_precision_score
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from data import load_data
from processing import get_preprocessor
from models import AdvancedXGBClassifier
import shap
import polars as pl



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
    
    # Extract booster from your custom wrapper
    if hasattr(model, 'booster_') and model.booster_ is not None:
        booster = model.booster_
    else:
        return

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # 1. Summary Dot Plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Champion Model)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close(fig)
    mlflow.log_artifact("shap_summary.png")

    # 2. Bar Plot
    fig_bar = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_bar.png")
    plt.close(fig_bar)
    mlflow.log_artifact("shap_bar.png")

    # Cleanup
    if os.path.exists("shap_summary.png"): os.remove("shap_summary.png")
    if os.path.exists("shap_bar.png"): os.remove("shap_bar.png")

def objective(trial, X, y):
    with mlflow.start_run(nested = True, run_name = f"Trial_{trial.number}"):

        obj = trial.suggest_categorical('objective_type', ['standard', 'kl', 'entropy', 'focal'])
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'n_jobs': -1
        }
        reg_alpha = 0.0
        gamma_ind = 0.0
        
        if obj == 'focal':
            gamma_ind = trial.suggest_float('gamma_ind', 0.5, 5.0)
        elif obj in ['kl', 'entropy']:
            reg_alpha = trial.suggest_float('reg_alpha', 0.1, 5.0)

        model = AdvancedXGBClassifier(
            objective_type=obj,
            reg_alpha=reg_alpha,
            gamma_ind=gamma_ind,
            **param_grid
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            auc_scores.append(average_precision_score(y_val, probs))
            
        mean_auc = np.mean(auc_scores)
        
        mlflow.log_params(param_grid)
        mlflow.log_param("objective", obj)
        mlflow.log_param("reg_alpha", reg_alpha)
        mlflow.log_param("gamma_ind", gamma_ind)
        mlflow.log_metric("cv_mean_pr_auc", mean_auc)
        
        return mean_auc

FILE_PATH = 'optimized_data.parquet'
#TARGET_COL = 'Class'
TARGET_COL = 'isFraud'
TEST_SIZE = 0.2
TRIALS = {
    'kl': [0.1,0.2,0.3, 0.5, 1.0, 2.0],
    'entropy': [0.1, 0.3, 0.5], 
    'focal': [1.0, 1.5, 2.0, 3.0] 
}

def main():
    mlflow.set_experiment("Fraud_Detection_Refactored")
    
    print("--- Loading Data ---")
    X_raw, y = load_data(FILE_PATH, TARGET_COL)
    
    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X_raw, y, test_size=0.15, stratify=y, random_state=42
    )
    
    print("--- Fitting Preprocessor ---")
    preprocessor = get_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_holdout = preprocessor.transform(X_holdout_raw).astype(np.float32)
    
    
    try:
        num_names = preprocessor.named_transformers_['num'].get_feature_names_out().tolist()
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
        feature_names = num_names + cat_names
    except:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    
    print("---Starting Optuna Run---")

    
    with mlflow.start_run(run_name="Master_Experiment") as parent_run:
        
        study = optuna.create_study(direction="maximize")
        # Pass X_train and y_train (pd.Series) to objective
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        print("--- Optimization Complete ---")
        best_trial = study.best_trial
        print(f"Best PR_AUC: {best_trial.value}")
        print(f"Best Params: {best_trial.params}")
        
        # Log Best Params to Parent Run
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_cv_pr_auc", best_trial.value)
        print("--- Retraining Champion Model ---")
        
        # Reconstruct params
        best_params = best_trial.params.copy()
        obj_type = best_params.pop('objective_type')
        reg_alpha = best_params.pop('reg_alpha', 0.0)
        gamma_ind = best_params.pop('gamma_ind', 0.0)
        
        champion_model = AdvancedXGBClassifier(
            objective_type=obj_type,
            reg_alpha=reg_alpha,
            gamma_ind=gamma_ind,
            **best_params
        )
        
        champion_model.fit(X_train, y_train)
        
        # 4. Final Evaluation on Holdout Set
        probs = champion_model.predict_proba(X_holdout)[:, 1]
        metrics = evaluate_metrics(y_holdout, probs)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})
        
        # 5. Generate SHAP (Only for Champion)
        X_shap_sample = X_train[:1000] # Sample for speed
        generate_shap_artifacts(champion_model, X_shap_sample, feature_names)
        
        # 6. Log Model to Registry
        signature = infer_signature(X_train[:5], champion_model.predict(X_train[:5]))
        
        # We use sklearn flavor because it's a wrapper class
        mlflow.sklearn.log_model(
            sk_model=champion_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="AdvancedFraudXGB"
        )
        
        print(f"Champion model registered as 'AdvancedFraudXGB'. Holdout PR-AUC: {metrics['PR_AUC']:.4f}")

if __name__ == "__main__":
    main()