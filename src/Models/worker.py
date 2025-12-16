import os
from celery import Celery
from db import SessionLocal, Dataset, MLModel
from data_engine import DataManager
from schemas import Features
from runner import ExperimentRunner
import traceback
import requests
from config import settings

REDIS_URL = settings.REDIS_URL

celery_app = Celery(
    "fraud_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

@celery_app.task(bind=True)
def run_user_pipeline(self, dataset_id: int, user_id: int, model_type="ensemble", n_trials=10):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"status": "failed", "error": "Dataset not found"}

        print(f"[User {user_id}] Starting Pipeline for Dataset {dataset_id}")
        schema = Features(**dataset.schema_map)
        handler = DataManager(schema)
        handler.process(dataset.storage_path)
        runner = ExperimentRunner(user_id = user_id, datahandler=handler)
        run_id = runner.run_experiment(model_name=model_type, n_trials=n_trials)
        db.query(MLModel).filter(MLModel.user_id == user_id).update({"status": "archived"})

        new_model = MLModel(
            user_id=user_id,
            dataset_id=dataset_id,
            model_type=model_type,
            mlflow_run_id=run_id,
            status="ready"
        )
        db.add(new_model)
        db.commit()
        print(f"[User {user_id}] Training complete. Run ID: {run_id}")
        try:
            requests.post(
           "http://api-service:8000/internal/reload-model",
            json={"user_id": user_id, "run_id": run_id},
            headers={"x-admin-key": settings.ADMIN_SECRET},
            timeout=5
        )
        except:
               print("Warning: Failed to notify API of model update")
        return {"status": "success", "run_id": run_id}
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()
