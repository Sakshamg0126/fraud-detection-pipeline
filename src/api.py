import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")


@lru_cache
def load_artifacts():
    model = joblib.load("models/model.joblib")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return model, meta["features"], meta["threshold"]


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    model, features, threshold = load_artifacts()
    missing = [f for f in features if f not in tx.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    X = pd.DataFrame([tx.features])[features]
    prob = float(model.predict_proba(X)[0, 1])
    return {"fraud_probability": prob, "is_fraud": prob >= threshold, "threshold": threshold}
