

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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.model_utils import evaluate_metrics, generate_shap_artifacts
import shap
import polars as pl
from spaces import SearchSpaceRegistry
from ModelFactory import ModelFactory


class ExperimentRunner:
    def __init__(self, experiment_name, data_path, target_col):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.target_col = target_col
        mlflow.set_experiment(experiment_name)
        self.X, self.y = load_data(data_path, target_col=target_col)
        counts = self.y.value_counts()
        self.neg_pos_ratio = counts[0] / counts[1]
        print(f"Calculated scale_pos_weight: {self.neg_pos_ratio:.2f}")
        self.cat_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = self.X.select_dtypes(include=['number']).columns.tolist()
    def objective(self, trial, model_name):
        with mlflow.start_run(nested = True, run_name = f"Trial_{trial.number}"):
            params = SearchSpaceRegistry.get_search_space(model_name=model_name,trial=trial)
            if model_name == "ensemble":
                params["cat_features"] = self.cat_cols
                params["num_features"] = self.num_cols
                params["xgb_scale_pos_weight"] = self.neg_pos_ratio
            elif model_name == "xgboost":
                params["scale_pos_weight"] = self.neg_pos_ratio
            model = ModelFactory.create_model(params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            auc_scores = []
        
            for train_idx, val_idx in cv.split(self.X, self.y):
               
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
                model.fit(X_tr, y_tr)
                try:
                    preds = model.predict_proba(X_val)[:, 1]
                except:
                    preds = model.predict(X_val)
                
                score = average_precision_score(y_val, preds)
                auc_scores.append(score)
            
            
            mean_auc = np.mean(auc_scores)
            clean_params = {k:v for k,v in params.items() if isinstance(v, (int, float, str))}
            mlflow.log_params(clean_params)
            mlflow.log_metric("cv_mean_pr_auc", mean_auc)
        
            return mean_auc
    def run_experiment(self, model_name, n_trials):
        print(f"--- Starting Optimization for {model_name} ---")
        with mlflow.start_run(run_name=f"HPO_{model_name}") as parent_run:
            study = optuna.create_study(direction="maximize")
            
            study.optimize(lambda trial: self.objective(trial, model_name), n_trials=n_trials)
            
            best_trial = study.best_trial
            print(f"Best AUPRC for {model_name}: {best_trial.value:.4f}")
            
            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metric("best_cv_score", best_trial.value)

if __name__ == "__main__":
    
    runner = ExperimentRunner(
        experiment_name="Modular_Fraud_System",
        data_path="creditcard.csv",
        target_col="Class"
    )
    
    runner.run_experiment("xgboost", n_trials=10)
    
    runner.run_experiment("catboost", n_trials=10)
    runner.run_experiment("ensemble", n_trials=10)