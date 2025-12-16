import mlflow.pyfunc
import os
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
from threading import RLock
import threading
from typing import Dict, Any
from db import SessionLocal, MLModel
from config import settings

class ModelManager:
    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        self._models: Dict[int, Any] = {}
        self._global_model = None
        self._cache = LRUCache(maxsize=settings.MODEL_CACHE_SIZE)
        self._lock = RLock()

        self.load_global_model()

    def load_global_model(self):
        try:
            print("Loading global model into memory..")
            self._global_model = mlflow.pyfunc.load_model(settings.GLOBAL_MODEL_URI)
            print("Finished loafing global model")
        except Exception as e:
            print(f"Failed to load global model: {e}")

    def load_model(self, user_id:int, run_id :int):
        if user_id in self._models and self._models[user_id]["run_id"] == run_id:
            return
        try:
            print(f"Loading model for {user_id} with run id {run_id}")
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/User_{self.user_id}_Model")
            with self._lock:

                self._models[user_id] = {"model": model,"run_id": run_id}

        except Exception as e:
            print(f"Error loading model for user {user_id}: {e}")

    def get_model(self, user_id):
        db = SessionLocal()
        try:
            record = db.query(MLModel).filter(MLModel.user_id == user_id, MLModel.status=="ready").order_by(MLModel.created_at.desc()).first()
        finally:
            db.close()
        if not record:
            return self._global_model, "global"
        run_id = record.mlflow_run_id
        cache_key = (user_id, run_id)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key], "custom"
        try:
            print(f"Loading model for {user_id} with run id {run_id}")
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/User_{self.user_id}_Model")
            with self._lock:
                self._cache[cache_key] = model
            return model, "custom"
        except Exception as e:
            print(f"Failed to load custom model for {user_id}: {e}")
            return self._global_model, "global_fallback"

    

    def get_prediction(self, user_id, input_df):
        model, source = self.get_model(user_id= user_id)
        
        try:
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(input_df)[:, 1][0]), source
            else:
                return float(model.predict(input_df)[0]), source
        except Exception as e:
            print(f"Inference error user {user_id}: {e}")
            return 0.0, "error"
        
model_manager = ModelManager()



    