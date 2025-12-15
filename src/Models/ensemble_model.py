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
from models import AdvancedXGBClassifier
import polars as pl
from Utils.model_utils import TimeSeriesValidator

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
        
        self.model_xgb = AdvancedXGBClassifier()

        self.meta_model = LogisticRegression()
    def _to_pandas(self, data):
      
        if data is None:
            return None
            
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
            
        if isinstance(data, (pl.DataFrame, pl.Series)):
            pdf = data.to_pandas()
        else:
            pdf = data
        if isinstance(pdf, pd.DataFrame):
            pdf = pdf.copy()
            
            for col in self.cat_features:
                if col in pdf.columns:
                    pdf[col] = pdf[col].astype(object).fillna("Missing").astype(str)
            for col in self.num_features:
                if col in pdf.columns:
                    pdf[col] = pd.to_numeric(pdf[col], errors='coerce').astype(float)
        
        return pdf

    def fit(self, X, Y):
        X = self._to_pandas(X)
        Y = self._to_pandas(Y)
        
        if isinstance(Y, pd.DataFrame):
            Y = Y.iloc[:, 0]
        
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        tscv = TimeSeriesValidator(n_splits=3)
        meta_preds_list = []
        meta_y_list = []
        
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
            
            self.model_cat.fit(X_tr, y_tr)
            preds_cat = self.model_cat.predict_proba(X_val)[:, 1]

            neg_count = (y_tr == 0).sum()
            pos_count = (y_tr == 1).sum()
            if pos_count == 0: weight = 1.0 
            else: weight = neg_count / pos_count
            
            self.model_xgb.set_params(scale_pos_weight=weight)
            
            cols_xgb = self.num_features if self.num_features else X_tr.columns
            cols_xgb = [c for c in cols_xgb if c in X_tr.columns]
            
            self.model_xgb.fit(X_tr[cols_xgb], y_tr)
            preds_xgb = self.model_xgb.predict_proba(X_val[cols_xgb])[:, 1]
            
            fold_meta_features = np.column_stack((preds_cat, preds_xgb))
            
            meta_preds_list.append(fold_meta_features)
            meta_y_list.append(y_val)
            
            print(f"Fold {fold+1} processed. Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        meta_X = np.vstack(meta_preds_list)
        meta_y = np.concatenate(meta_y_list)
        
        print(f"Training Meta-Learner on {len(meta_y)} samples...")
        self.meta_model.fit(meta_X, meta_y)
        print("Retraining Base Models on full data")
        self.model_cat.fit(X, Y)
        
        cols_xgb = self.num_features if self.num_features else X.columns
        cols_xgb = [c for c in cols_xgb if c in X.columns]
        self.model_xgb.fit(X[cols_xgb], Y)
        
        self.ensemble_fitted = True
        return self
    def predict(self, X):
        if not self.ensemble_fitted: raise ValueError("Model not fitted")
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X_cat = self._to_pandas(X)
        p_cat = self.model_cat.predict_proba(X_cat)[:, 1]
        p_xgb = self.model_xgb.predict_proba(X[self.num_features])[:, 1]
        
        stacked_input = np.column_stack((p_cat, p_xgb))
        
        return self.meta_model.predict_proba(stacked_input)
    def get_xgboost_booster(self):
        """Returns the underlying XGBoost booster object"""
        if not hasattr(self.model_xgb, "fit"): 
             raise ValueError("Model not fitted yet")
       
        return self.model_xgb.get_booster()

    def get_catboost_booster(self):
        """Returns the underlying CatBoost object"""
        return self.model_cat

    def get_meta_weights(self):
        """Returns how much the ensemble trusts CatBoost vs XGBoost"""
      
        return self.meta_model.coef_[0]
    

