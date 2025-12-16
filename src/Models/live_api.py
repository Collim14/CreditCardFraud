from fastapi import FastAPI, Depends, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import json
from sqlalchemy.orm import Session
import pandas as pd
from inference import model_manager
from features import RealTimeFeatureEngine
from db import get_db, User
from config import settings
import redis

app = FastAPI()
redis_client = redis.from_url(settings.REDIS_URL)

def get_user_from_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    cached_id = redis_client.get(f"auth:{x_api_key}")
    if cached_id:
        return int(cached_id)
    user = db.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    redis_client.setex(f"auth:{x_api_key}", 300, user.id)
    return user.id


@app.on_event("startup")
def startup_event():
   
    print("Server Startup: Warming up Models")
    model_manager.warmup_cache()
    print("Server Ready to Process Payments")

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    timestamp: float
    metadata: Dict[str, Any] 

class FraudResponse(BaseModel):
    decision: str
    score: float
    latency_ms: float
    source: str

class ReloadRequest(BaseModel):
    user_id:int
    run_id:int

@app.post("/internal/reload-model")
def reload_model_hook(payload: ReloadRequest, x_admin_key: str = Header(...)):
    if x_admin_key != settings.ADMIN_SECRET:
        raise HTTPException(status_code=403)
    model_manager.load_model(user_id=payload.user_id, run_id=payload.run_id)
    return {"status": "reloaded"}


def log_transaction(user_id:int, request_data:dict, prediction: dict):
    log_entry = {
        "timestamp": time.time(),
        "user_id": user_id,
        "input": request_data,
        "output": prediction
    }
    pass

@app.post("/v1/assess", response_model=FraudResponse)
async def assess_transaction(
    txn: TransactionRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_user_from_key)
):
    start_time = time.time()
    
    features = txn.metadata.copy()
    features["TransactionAmt"] = txn.amount
    features["TransactionDT"] = txn.timestamp
    features_pd = pd.DataFrame([features])

    
    score, mtype = model_manager.get_prediction(user_id=user_id, input_df=features_pd)
    

    decision = "ALLOW"
    if score > 0.9:
        decision = "BLOCK"
    elif score > 0.7:
        decision = "REVIEW"
    latency = (time.time() - start_time) * 1000
    background_tasks.add_task(
        log_transaction, 
        user_id, 
        features, 
        {"score": score, "decision": decision}
    )
    

    return {
        "decision": decision,
        "core": score,
        "latency_ms": latency,
        "reason": "High velocity detected" if score > 0.9 else "Normal behavior"
    }

