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
                    pdf[col] = pdf[col].astype(str).fillna("Missing")
              
                    pdf[col] = pdf[col].astype("category")
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

        
        
        
        
        self.model_cat.fit(X, Y)
        preds_cat = self.model_cat.predict_proba(X)[:, 1]

        neg_count = (Y == 0).sum()
        pos_count = (X == 1).sum()
        if pos_count == 0: weight = 1.0 
        else: weight = neg_count / pos_count
            
        self.model_xgb.set_params(scale_pos_weight=weight)
            
            
        self.model_xgb.fit(X, Y)
        preds_xgb = self.model_xgb.predict_proba(X_val)[:, 1]
            
        meta_X = np.column_stack((preds_cat, preds_xgb))
        
        print(f"Training Meta-Learner on {len(Y)} samples...")
        self.meta_model.fit(meta_X, Y)
        
        self.ensemble_fitted = True
        return self
    
    def predict(self, X):
        if not self.ensemble_fitted: raise ValueError("Model not fitted")
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X_cat = self._to_pandas(X)
        p_cat = self.model_cat.predict_proba(X_cat)[:, 1]
        p_xgb = self.model_xgb.predict_proba(X)[:, 1]
        
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
    

