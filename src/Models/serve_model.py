import mlflow.pyfunc
import pandas as pd
import os
import shap
import numpy as np
class FraudPredictor:
    def __init__(self, model_name = "Production_Model", stage = "latest"):
        self.model_name = model_name
        self.stage = stage
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Loading the model{model_name}, at stage {stage}")
        try:
            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"Sucessfully loaded model {model_uri}")
        except Exception as e:
            print(f"Model failed to load {e}")

            raise e
        self.explainer = shap.TreeExplainer(self.model) 
        
    def predict(self, X: pd.DataFrame):

        return self.model.predict(X)
    def predict_with_shap(self, X:pd.DataFrame, num_features = 3):
        probs = self.model.predict_proba(X)[:, 1]
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        abs_shap = np.abs(shap_values)
    
        top_indices = np.argsort(-abs_shap, axis=1)[:, :num_features]
        features_arr = X.columns.to_numpy()[top_indices] 
        impacts_arr = np.take_along_axis(shap_values, top_indices, axis=1)
        values_arr = np.take_along_axis(X.to_numpy(), top_indices, axis=1)

        results = [
        {
                "fraud_probability": float(probs[i]),
                "shap_codes": [
                    {
                        "feature": features_arr[i, j],
                        "value": float(values_arr[i, j]),
                        "impact": float(impacts_arr[i, j])
                    }
                    for j in range(num_features)
                ]
            }
            for i in range(len(X))
        ]
            
        return results

