import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Literal, Tuple, List, Optional




class AdvancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom XGBoost Wrapper supporting:
    1. Standard LogLoss
    2. KL-Divergence Regularization (Weighted LogLoss)
    3. Entropy Regularization (Penalize confidence)
    4. Focal Loss (Focus on hard/information-rich examples)
    """
    def __init__(self, 
                 objective_type: Literal['standard', 'kl', 'entropy', 'focal'] = 'standard',
                 reg_alpha: float = 0.0, 
                 gamma_ind: float = 2.0, 
                 **xgb_params):
        """
        Args:
            objective_type: The type of custom objective.
            reg_alpha (float): Weight for KL or Entropy regularization (lambda).
            gamma_ind (float): The focusing parameter for Focal Loss.
            **xgb_params: Standard XGBoost parameters (max_depth, eta, etc.)
        """
        self.objective_type = objective_type
        self.reg_alpha = reg_alpha
        self.gamma_ind = gamma_ind
        self.xgb_params = xgb_params
        self.booster_ = None
        
        if 'objective' not in self.xgb_params:
            self.xgb_params['objective'] = 'binary:logistic'

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _custom_objective(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Gradient and Hessian based on the selected strategy.
        Note: 'preds' in custom obj are raw log-odds (margins), not probabilities.
        """
        labels = dtrain.get_label()
        probs = self._sigmoid(preds)
        
        # KL Divergence / Weighted LogLoss
        # L = (1 + alpha) * LogLoss
        if self.objective_type == 'kl':
            weight = 1.0 + self.reg_alpha
            grad = weight * (probs - labels)
            hess = weight * probs * (1.0 - probs)

        # Entropy Regularization
        # L = LogLoss + alpha * Entropy(p)
        # Encourages higher entropy (less confidence), good for noisy fraud labels.
        elif self.objective_type == 'entropy':
            # Gradient of Entropy term: alpha * p * (1-p) * preds
            grad = (probs - labels) - self.reg_alpha * probs * (1.0 - probs) * preds
            hess = probs * (1.0 - probs) * (1.0 - self.reg_alpha - self.reg_alpha * (1.0 - 2.0 * probs) * preds)

        # Focal Loss
        # L = -alpha * (1-p_t)^gamma * log(p_t)
        # Focuses gradients on samples the model is currently getting WRONG.
        # This effectively replaces the "EntropyGain" logic by focusing on high-info samples.
        elif self.objective_type == 'focal':
            gamma = self.gamma_ind
            # Compute p_t: probability of the true class
            p_t = labels * probs + (1 - labels) * (1 - probs)
            
            # First order derivative (Gradient)
            # This formula is complex, standard implementation below:
            modulator = (1 - p_t) ** gamma
            term1 = gamma * (1 - p_t)**(gamma - 1) * p_t * np.log(p_t + 1e-9)
            grad = modulator * (probs - labels) + term1 * (labels * (1-probs) - (1-labels)*probs)
            
            # Hessian Approximation (Exact Hessian is unstable for Focal Loss)
            # We use the upper bound approximation often used in LightGBM/XGB implementations
            hess = (1 - p_t) ** gamma * (probs * (1 - probs))
            hess = np.maximum(hess, 1e-16) # Stability

        else:
            raise ValueError(f"Objective type '{self.objective_type}' should be handled natively in fit(), not here.")

        return grad, hess

    def fit(self, X, y, eval_set=None, verbose=False):
        dtrain = xgb.DMatrix(X, label=y)
        
        params = self.xgb_params.copy()
        num_rounds = params.pop('n_estimators', 100)
        
        # Prepare Watchlist
        watchlist = [(dtrain, 'train')]
        if eval_set:
            for i, (ex, ey) in enumerate(eval_set):
                watchlist.append((xgb.DMatrix(ex, label=ey), f'eval_{i}'))

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
        dtest = xgb.DMatrix(X)
        # predict returns raw probs because we set binary:logistic, 
        # even with custom objective, unless we return margin.
        probs = self.booster_.predict(dtest)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        if not self.booster_: raise ValueError("Model not fitted")
        dtest = xgb.DMatrix(X)
        pos_probs = self.booster_.predict(dtest)
        return np.vstack((1 - pos_probs, pos_probs)).T
    def get_booster(self):
        if not self.booster_: raise ValueError("Model not fitted")
        return self.booster_