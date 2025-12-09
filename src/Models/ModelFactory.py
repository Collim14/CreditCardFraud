from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from models import AdvancedXGBClassifier
from ensemble_model import EnsembleModel

class ModelFactory:
    @staticmethod
    def create_model(params:dict):
        p = params.copy()
        model = p.pop("model_type")
        if model == "xgboost":
            obj = p.pop("objective_type", "standard")
            gamma = p.pop("gamma_ind", 0.0)
            return AdvancedXGBClassifier(objective_type=obj, gamma_ind=gamma, **p)
        elif model == "catboost":
            return CatBoostClassifier(**p)
        elif model == "ensemble":
            cat_features = p.pop("cat_features", [])
            num_features = p.pop("num_features",[])
            xgb_params = {k.replace("xgb_", ""): v for k, v in p.items() if k.startswith("xgb_")}
            cat_params = {k.replace("cat_", ""): v for k, v in p.items() if k.startswith("cat_")}
            meta_params = {k.replace("meta_", ""): v for k, v in p.items() if k.startswith("meta_")}

            ensemble = EnsembleModel(cat_features=cat_features, num_features=num_features)
            ensemble.model_xgb.set_params(**xgb_params)
            ensemble.model_cat.set_params(**cat_params)
            if "C" in meta_params:
                ensemble.meta_model.set_params(C=meta_params["C"])
            return ensemble
        else:
            raise ValueError(f"Factory cannot create: {model}")