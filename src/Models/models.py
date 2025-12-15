import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Literal, Tuple, List, Optional




class AdvancedXGBClassifier(BaseEstimator, ClassifierMixin):
 
    def __init__(self, 
                 objective_type: Literal['standard', 'kl', 'entropy', 'focal'] = 'standard',
                 reg_alpha: float = 0.0, 
                 gamma_ind: float = 2.0, 
                 **xgb_params):
       
        self.objective_type = objective_type
        self.reg_alpha = reg_alpha
        self.gamma_ind = gamma_ind
        self.xgb_params = xgb_params
        self.booster_ = None
        
        if 'objective' not in self.xgb_params:
            self.xgb_params['objective'] = 'binary:logistic'
        if 'device' not in self.xgb_params:
            self.xgb_params['device'] = 'cpu'

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _custom_objective(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        
        labels = dtrain.get_label()
        probs = self._sigmoid(preds)
        
        # KL Divergence
        if self.objective_type == 'kl':
            weight = 1.0 + self.reg_alpha
            grad = weight * (probs - labels)
            hess = weight * probs * (1.0 - probs)

        # Entropy Regularisation
        elif self.objective_type == 'entropy':
            grad = (probs - labels) - self.reg_alpha * probs * (1.0 - probs) * preds
            hess = probs * (1.0 - probs) * (1.0 - self.reg_alpha - self.reg_alpha * (1.0 - 2.0 * probs) * preds)

        elif self.objective_type == 'focal':
            gamma = self.gamma_ind
            p_t = labels * probs + (1 - labels) * (1 - probs)
            
            modulator = (1 - p_t) ** gamma
            term1 = gamma * (1 - p_t)**(gamma - 1) * p_t * np.log(p_t + 1e-9)
            grad = modulator * (probs - labels) + term1 * (labels * (1-probs) - (1-labels)*probs)
            
            hess = (1 - p_t) ** gamma * (probs * (1 - probs))
            hess = np.maximum(hess, 1e-16)

        else:
            raise ValueError(f"Objective type '{self.objective_type}' should be handled natively in fit(), not here.")

        return grad, hess

    def fit(self, X, y, eval_set=None, verbose=False):
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
     
        dtrain = xgb.DMatrix(X, label=y, enable_categorical = True, )
        
        params = self.xgb_params.copy()
        num_rounds = params.pop('n_estimators', 100)
        
        watchlist = [(dtrain, 'train')]
        if eval_set:
            for i, (ex, ey) in enumerate(eval_set):
                watchlist.append((xgb.DMatrix(ex, label=ey, enable_categorical = True), f'eval_{i}'))

        obj_func = self._custom_objective if self.objective_type != 'standard' else None

        self.booster_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            obj=obj_func,
            evals=watchlist,
            verbose_eval=verbose
        )
        return self

    def predict(self, X):
        if not self.booster_: raise ValueError("Model not fitted")
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        dtest = xgb.DMatrix(X, enable_categorical = True)
        probs = self.booster_.predict(dtest)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        if not self.booster_: raise ValueError("Model not fitted")
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        dtest = xgb.DMatrix(X, enable_categorical = True)
        pos_probs = self.booster_.predict(dtest)
        return np.vstack((1 - pos_probs, pos_probs)).T
    def get_booster(self):
        if not self.booster_: raise ValueError("Model not fitted")
        return self.booster_
    
    def set_params(self, **params):
       
        for p in list(params.keys()):
            if hasattr(self, p):
                setattr(self, p, params.pop(p, None))
    
        self.xgb_params.update(params)
        