"""FastAPI wrapper serving the model registered as telco-churn-classifier@production.
Run: uv run uvicorn src.serve:app --port 8001"""
from pathlib import Path
import mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
MODEL_URI = "models:/telco-churn-classifier@production"
model = mlflow.sklearn.load_model(MODEL_URI)
app = FastAPI(title="Telco churn model (registry: production)")

class Batch(BaseModel):
    rows: list[dict]

@app.get("/health")
def health(): return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(b: Batch):
    X = pd.DataFrame(b.rows)
    p = model.predict_proba(X)[:, 1]
    return {"churn_probability": p.round(4).tolist(), "churn": (p >= 0.5).astype(int).tolist()}
