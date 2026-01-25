from fastapi import FastAPI
from app.predict import predict, predict_and_explain
from app.schemas import PredictionRequest, Prediction, PredictionAndExplanation
from pathlib import Path
import joblib

app = FastAPI(title="ML Spam Detector API")
MODEL_PATH = Path("artifacts/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictionAndExplanation)
def predict(request: PredictionRequest):
    return predict_and_explain(model, request.message)