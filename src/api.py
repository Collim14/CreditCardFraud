import time
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from prometheus_client import Counter, Histogram, make_asgi_app
from scipy.stats import ks_2samp

from Models.serve_model import FraudPredictor

class TransactionInput(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    meta: Dict[str, Any]



REQUEST_COUNT = Counter("fraud_req_total", "Total fraud detection requests")
FRAUD_DETECTED = Counter("fraud_detected_total", "Count of positive fraud predictions")
LATENCY = Histogram("fraud_req_latency_seconds", "Time spent processing request")
DRIFT_ALERTS = Counter("data_drift_alerts", "Count of detected data drift events")

app = FastAPI(title="Fraud Detection API")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


predictor: FraudPredictor = None
BASELINE_DATA: Optional[pd.DataFrame] = None
DRIFT_BUFFER: List[Dict] = []
DRIFT_BUFFER_SIZE = 100


def load_baseline_data():
    """
    Yet to fully implement, mock version currently in place
    """
    try:
        global BASELINE_DATA
        BASELINE_DATA = pd.DataFrame({
            "TransactionAmt": np.random.uniform(10, 1000, 1000),
            "V1": np.random.normal(0, 1, 1000)
        })
        print("Baseline loaded for detecting drift ")
    except Exception as e:
        print(f"Failed to load baseline {e}")

def check_data_drift(new_batch: List[Dict]):
    """
    Calculates if incoming transaction belongs to known distribution using simple ks test, not
    fully implemented atm
    """
    if BASELINE_DATA is None or len(new_batch) == 0:
        return

    try:
        new_df = pd.DataFrame(new_batch)
        drift_detected = False
        #Add key features for data
        key_features = []
        
        for feature in key_features:
            if feature in BASELINE_DATA.columns:
                stat, p_value = ks_2samp(BASELINE_DATA[feature], new_df[feature])
                if p_value < 0.01:
                    print(f"WARNING: Drift detected for feature {feature}, with p={p_value:.5f}")
                    drift_detected = True
        
        if drift_detected:
            DRIFT_ALERTS.inc()
            
    except Exception as e:
        print(f"Error with checking for data drift: {e}")

def get_dynamic_threshold(transaction_amt: float) -> float:
    """
    Sets threshold depending on transaction value
    """
    if transaction_amt is None or transaction_amt == 0:
        return 0.5
    cost_fp = 5.0
    return cost_fp / (cost_fp + transaction_amt)

@app.middleware("http")
async def monitor_performance(request: Request, call_next):
    """Tracks latency for every request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    LATENCY.observe(process_time)
    return response


@app.on_event("startup")
def load_model_on_startup():
    global predictor
    try:
        predictor = FraudPredictor()
        load_baseline_data()
        print("Model loaded")
    except Exception as e:
        print("Model failed to load")
        print(e)

    
@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(transaction: TransactionInput, background_tasks: BackgroundTasks):
    global predictor, DRIFT_BUFFER
    
    REQUEST_COUNT.inc()
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        features = transaction.features
        input_df = pd.DataFrame([features])
        
        obj_cols = input_df.select_dtypes(include=['object']).columns
        input_df[obj_cols] = input_df[obj_cols].astype('category')
        if hasattr(predictor.model, "predict_proba"):
            pred_probs = predictor.model.predict_proba(input_df)
            fraud_prob = float(pred_probs[0][1])
        else:
            pred = predictor.predict(input_df)
            fraud_prob = float(pred[0])

        amount = features.get("TransactionAmt", 0.0)
        
        threshold = 0.5
        if amount > 0:
            threshold = get_dynamic_threshold(fraud_prob, amount)
        
        is_fraud = fraud_prob > threshold

        if is_fraud:
            FRAUD_DETECTED.inc()
        DRIFT_BUFFER.append(features)
        if len(DRIFT_BUFFER) >= DRIFT_BUFFER_SIZE:
            batch_to_check = DRIFT_BUFFER.copy()
            DRIFT_BUFFER.clear()
            background_tasks.add_task(check_data_drift, batch_to_check)

        return {
            "fraud_probability": fraud_prob,
            "is_fraud": bool(is_fraud),
            "threshold_used": float(threshold),
            "meta": {
                "transaction_amount": amount,
                "model_version": predictor.stage if hasattr(predictor, 'stage') else "unknown"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model Loading")
    return {"status": "active", "model": "loaded"}
