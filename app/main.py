from fastapi import FastAPI
from app.model import load_model
from app.predict import predict, predict_and_explain
from app.schemas import PredictionRequest, Prediction, PredictionAndExplanation

app = FastAPI(title="Spam Detector API")

model = load_model()

@app.post("/predict", response_model=PredictionAndExplanation)
def predict(request: PredictionRequest):
    return predict_and_explain(model, request.message)