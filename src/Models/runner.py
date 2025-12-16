

import pandas as pd
import polars as pl
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

from data import DataHandler
from models import AdvancedXGBClassifier
import sys
from Utils.model_utils import evaluate_metrics, generate_shap_artifacts, TimeSeriesValidator
import shap
import polars as pl
from spaces import SearchSpaceRegistry
from ModelFactory import ModelFactory


class ExperimentRunner:
    def __init__(self, user_id: int, datahandler):
        self.user_id = user_id
        self.experiment_name = f"tenant_{user_id}_fraud_detection"
        mlflow.set_experiment(self.experiment_name)
        self.datah = datahandler

        counts = self.datah.df.select([
            (pl.col(self.datah.target_col) == 0).sum().alias("neg"),
            (pl.col(self.datah.target_col) == 1).sum().alias("pos")
        ]).collect()

        neg_count, pos_count = counts["neg"][0],counts["pos"][0]
        
        self.neg_pos_ratio = neg_count / pos_count if pos_count > 0 else 1.0

        
        
       
        

    def objective(self, trial, model_name, X_sample, y_sample):
        with mlflow.start_run(nested = True, run_name = f"Trial_{trial.number}"):
            params = SearchSpaceRegistry.get_search_space(model_name=model_name,trial=trial)
            neg_local = (y_sample == 0).sum()
            pos_local = (y_sample == 1).sum()
            local_ratio = neg_local / pos_local
            if model_name == "ensemble":
                params["cat_features"] = self.datah.cat_features
                params["num_features"] = self.datah.num_features
                params["xgb_scale_pos_weight"] = local_ratio
                params['xgb_enable_categorical'] = True
            elif model_name == "xgboost":
                params['enable_categorical'] = True
                params["scale_pos_weight"] = local_ratio
            model = ModelFactory.create_model(params)
            cv = TimeSeriesValidator(n_splits=3)
            auc_scores = []
            X_pd = X_sample.to_pandas() if isinstance(X_sample, pl.DataFrame) else X_sample
            y_pd = y_sample.to_pandas() if isinstance(y_sample, pl.Series) else y_sample
            
        
            for train_idx, val_idx in cv.split(X_pd):
               
                X_tr, X_val = X_pd.iloc[train_idx], X_pd.iloc[val_idx]
                y_tr, y_val = y_pd.iloc[train_idx], y_pd.iloc[val_idx]
                
                model.fit(X_tr, y_tr)
                
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    preds = model.predict(X_val)
                
                score = average_precision_score(y_val, preds)
                auc_scores.append(score)
            
            
            mean_auc = np.mean(auc_scores)
            clean_params = {k:v for k,v in params.items() if isinstance(v, (int, float, str))}
            mlflow.log_params(clean_params)
            mlflow.log_metric("cv_mean_pr_auc", mean_auc)
        
            return mean_auc
    def run_experiment(self, model_name, n_trials):
        print(f"Starting Optimisation for {model_name}")
        with mlflow.start_run(run_name=f"HPO_{model_name}") as parent_run:
            df_sample = self.datah.get_hpo_sample(frac=0.1)
            y_sample = df_sample[self.datah.target_col]
            X_sample = df_sample.drop(self.datah.target_col)
            study = optuna.create_study(direction="maximize")
            
            study.optimize(lambda trial: self.objective(trial, model_name, X_sample, y_sample), n_trials=n_trials)
            
            best_trial = study.best_trial
            print(f"Best AUPRC for {model_name}: {best_trial.value:.4f}")
           
            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metric("best_cv_score", best_trial.value)
            mlflow.log_artifacts()
        with mlflow.start_run(run_name=f"Final_Best_{model_name}") as run:
            print(f"Training Final Model for User {self.user_id}")
            full_df = self.datah.get_full_data()
            y_full = full_df[self.datah.target_col]
            X_full = full_df.drop(self.datah.target_col)
            X_pd = X_full.to_pandas()
            best_params = best_trial.params
            if model_name == "ensemble":
                best_params["cat_features"] = self.datah.cat_features
                best_params["num_features"] = self.datah.num_features
                best_params["xgb_scale_pos_weight"] = self.neg_pos_ratio
                best_params['xgb_enable_categorical'] = True
            elif model_name == "xgboost":
                best_params['enable_categorical'] = True
                best_params["scale_pos_weight"] = self.neg_pos_ratio
            final_model = ModelFactory.create_model(best_params)
            
            final_model.fit(X_pd, y_full.to_pandas())
            
            predictions = final_model.predict(X_pd.iloc[:5])
            signature = infer_signature(X_pd.iloc[:5], predictions)
            
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"User_{self.user_id}_Model"
            )
            
            mlflow.log_params(best_params)
            mlflow.log_metric("final_cv_score", best_trial.value)
            
            return run.info.run_id