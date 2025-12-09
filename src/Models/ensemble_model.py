from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import mlflow

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_features = None, num_features = None):
        self.cat_features = cat_features
        self.num_features = num_features
        self.booster_ = None
        self.ensemble_fitted = False

        self.model_cat = CatBoostClassifier(
            iterations=150, 
            depth=6, 
            cat_features=self.cat_features,
            verbose=0,
            allow_writing_files=False,
            auto_class_weights='Balanced'
        )
        
        # XGBoost: Optimized for numerical data (Your Custom Objective could go here)
        self.model_xgb = XGBClassifier(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.1, 
            eval_metric='logloss',
            n_jobs=-1,
            scale_pos_weight=99 # Handle imbalance
        )

        self.meta_model = LogisticRegression()

    def fit(self, X, Y):
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 61)
        meta_features = np.zeros((X.shape[0],2))
        print("Starting Stacking Process")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X,Y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = Y.iloc[train_idx]
            self.model_cat.fit(X_tr, y_tr)
            preds_cat = self.model_cat.predict_proba(X_val)[:, 1]
            
            # --- Train XGBoost on Numerical Subset ---
            # XGBoost only gets numerical columns
            self.model_xgb.fit(X_tr[self.num_features], y_tr)
            preds_xgb = self.model_xgb.predict_proba(X_val[self.num_features])[:, 1]
            
            # Fill Meta Features
            meta_features[val_idx, 0] = preds_cat
            meta_features[val_idx, 1] = preds_xgb
            
            print(f"Fold {fold+1} processed.")
        # 2. Train Meta-Learner on the OOF predictions
        print("Training Meta-Learner...")
        self.meta_model.fit(meta_features, Y)
        
        # 3. Retrain Base Models on FULL Data for final inference
        print("Retraining Base Models on Full Data...")
        self.model_cat.fit(X, Y)
        self.model_xgb.fit(X[self.num_features], Y)
        self.ensemble_fitted = True
        
        return self
    def predict(self, X):
        if not self.ensemble_fitted: raise ValueError("Model not fitted")
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # 1. Get Base Predictions
        p_cat = self.model_cat.predict_proba(X)[:, 1]
        p_xgb = self.model_xgb.predict_proba(X[self.num_features])[:, 1]
        
        # 2. Stack them
        stacked_input = np.column_stack((p_cat, p_xgb))
        
        # 3. Final Prediction
        return self.meta_model.predict_proba(stacked_input)
    def get_xgboost_booster(self):
        """Returns the underlying XGBoost booster object"""
        if not hasattr(self.model_xgb, "fit"): 
             raise ValueError("Model not fitted yet")
        # XGBoost Sklearn API exposes .get_booster()
        return self.model_xgb.get_booster()

    def get_catboost_booster(self):
        """Returns the underlying CatBoost object"""
        return self.model_cat

    def get_meta_weights(self):
        """Returns how much the ensemble trusts CatBoost vs XGBoost"""
        # Returns [Weight_CatBoost, Weight_XGBoost]
        return self.meta_model.coef_[0]
    
def run_ensemble_training():
    mlflow.set_experiment("Ensemble_Fraud_Detection")
    
    # Simulate Data (Mixed Types)
    df = pd.DataFrame(np.random.randn(1000, 10), columns=[f"V{i}" for i in range(10)])
    # Add fake categorical columns
    df['Merchant'] = np.random.choice(['A', 'B', 'C', 'D'], 1000)
    df['Zip'] = np.random.choice(['10001', '90210', '33101'], 1000)
    # Target
    y = pd.DataFrame(np.random.randint(0, 2, 1000))
    
    cat_cols = ['Merchant', 'Zip']
    num_cols = [f"V{i}" for i in range(10)]
    
    with mlflow.start_run(run_name="Hybrid_Stacking"):
        
        ensemble = EnsembleModel(cat_features=cat_cols, num_features=num_cols)
        
        # Fit
        ensemble.fit(df, y)
        
        # Evaluate (In real life, use a holdout set here)
        probs = ensemble.predict_proba(df)[:, 1]
        score = average_precision_score(y, probs)
        
        # Log Metrics
        mlflow.log_metric("training_auprc", score)
        
        # Log Parameters
        mlflow.log_param("meta_learner", "LogisticRegression")
        mlflow.log_param("base_model_1", "CatBoost")
        mlflow.log_param("base_model_2", "XGBoost")
        
        # Log the Coefficients of the Meta Learner
        # This tells you which model is doing the heavy lifting
        weights = ensemble.meta_model.coef_[0]
        mlflow.log_metric("weight_catboost", weights[0])
        mlflow.log_metric("weight_xgboost", weights[1])
        
        print(f"Ensemble Training Complete. Meta-Weights: Cat={weights[0]:.2f}, XGB={weights[1]:.2f}")
        
        # Log the Model
        # Note: Custom models require 'python_model' flavor usually, 
        # but since we inherit from sklearn BaseEstimator, we can try sklearn flavor
        # or pickle it manually if complex.
        mlflow.sklearn.log_model(ensemble, "ensemble_model")

if __name__ == "__main__":
    run_ensemble_training()