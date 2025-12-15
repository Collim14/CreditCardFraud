

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.model_utils import evaluate_metrics, generate_shap_artifacts, TimeSeriesValidator
import shap
import polars as pl
from spaces import SearchSpaceRegistry
from ModelFactory import ModelFactory


class ExperimentRunner:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.datah = DataHandler()

        self.datah.process(data_path, target_col=target_col)
        counts = self.datah.df.select([
            (pl.col(self.datah.target_col) == 0).sum().alias("neg"),
            (pl.col(self.datah.target_col) == 1).sum().alias("pos")
        ]).collect()

        neg_count = counts["neg"][0]
        pos_count = counts["pos"][0]
        
        self.neg_pos_ratio = neg_count / pos_count
        
       
        

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
        
            for train_idx, val_idx in cv.split(X_sample):
               
                X_tr, X_val = X_sample[train_idx], X_sample[val_idx]
                y_tr, y_val = y_sample[train_idx], y_sample[val_idx]
            
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
        print(f"--- Starting Optimisation for {model_name} ---")
        with mlflow.start_run(run_name=f"HPO_{model_name}") as parent_run:
            df_sample = self.datah.get_hpo_sample(time_col= 'TransactionDT', frac=0.1)
            y_sample = df_sample.drop_in_place(self.datah.target_col)
            X_sample = df_sample
            study = optuna.create_study(direction="maximize")
            
            study.optimize(lambda trial: self.objective(trial, model_name, X_sample, y_sample), n_trials=n_trials)
            
            best_trial = study.best_trial
            print(f"Best AUPRC for {model_name}: {best_trial.value:.4f}")
           
            mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
            mlflow.log_metric("best_cv_score", best_trial.value)
            mlflow.log_artifacts()
        with mlflow.start_run(run_name=f"Final_Best_{model_name}"):
            full_df = self.datah.df.collect() 
            y_full = full_df.drop_in_place(self.datah.target_col)
            X_full = full_df
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
            final_model.fit(X_full, y_full)
            predictions = final_model.predict(X_full[:5])
            signature = infer_signature(X_full[:5].to_pandas(), predictions)
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                registered_model_name="Production_Model" 
            
            )
            print("----Final Production Model Logged----")



if __name__ == "__main__":
    data_path="~/Desktop/Credit-Card-Fraud/src/Features/transformed_IEEE.parquet"
    target_col="isFraud"
    datah = DataHandler()

    datah.process(data_path, target_col=target_col)
    runner = ExperimentRunner(
        experiment_name="Modular_Fraud_System"
    )
    
    runner.run_experiment("ensemble", n_trials=10)
    
    runner.run_experiment("catboost", n_trials=10)

    runner.run_experiment("ensemble", n_trials=10)