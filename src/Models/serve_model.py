import mlflow.pyfunc
import pandas as pd
import os

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
        
    def predict(self, X: pd.DataFrame):

        return self.model.predict(X)

