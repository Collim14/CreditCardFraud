import optuna

class SearchSpaceRegistry: 
    @staticmethod
    def get_xgboost_space(trial: optuna.Trial):
        return {
            "model_type": "xgboost",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "objective_type": trial.suggest_categorical("objective_type", ["standard", "focal", "kl", "entropy"]),
            "gamma_ind": trial.suggest_float("gamma_ind", 0.5, 3.0) if trial.params.get("objective_type") == "focal" else 0.0
        }

    @staticmethod
    def get_catboost_space(trial: optuna.Trial):
        return {
            "model_type": "catboost",
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "auto_class_weights": "Balanced"
        }
    @staticmethod
    def get_ensemble_space(trial: optuna.Trial):
        """
        Optimise params for both models simultaneously.
        """
        return {
            "model_type": "ensemble",
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.2),
            "xgb_objective_type": trial.suggest_categorical("objective_type", ["standard", "focal", "kl", "entropy"]),
            "xgb_gamma_ind": trial.suggest_float("gamma_ind", 0.5, 3.0) if trial.params.get("xgb_objective_type") == "focal" else 0.0,
            "cat_iterations": trial.suggest_int("cat_iterations", 100, 300),
            "cat_depth": trial.suggest_int("cat_depth", 4, 8),
            "cat_learning_rate": trial.suggest_float("cat_learning_rate", 0.05, 0.2),
            "cat_l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "meta_C": trial.suggest_float("meta_C", 0.1, 10.0)
        }

    @staticmethod
    def get_search_space(model_name: str, trial: optuna.Trial):
        if model_name == "xgboost":
            return SearchSpaceRegistry.get_xgboost_space(trial)
        elif model_name == "catboost":
            return SearchSpaceRegistry.get_catboost_space(trial)
        elif model_name =="ensemble":
            return SearchSpaceRegistry.get_ensemble_space(trial)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
    