from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import uvicorn

from Models.serve_model import FraudPredictor

class TransactionInput(BaseModel):
    features: Dict[str, Any]

app = FastAPI(title="Fraud Detection API")

predictor: FraudPredictor = None

@app.on_event("startup")
def load_model_on_startup():
    global predictor
    try:
        predictor = FraudPredictor()
    except Exception as e:
        print("Model failed to load")
        print(e)

    
@app.post("/predict")
def predict_transaction(transaction: TransactionInput):
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model to predict from")
    try:
        input = pd.DataFrame([transaction.features])
        prediction = predictor.predict(input)
        result = float(prediction[0]) if hasattr(prediction, "__getitem__") else float(prediction)
        
        return {
            "fraud_probability": result,
            "alert": result > 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
