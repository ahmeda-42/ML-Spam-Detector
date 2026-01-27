from fastapi import FastAPI
from model.predict import predict, predict_and_explain
from app.schemas import PredictionRequest, Prediction, PredictionAndExplanation

app = FastAPI(title="ML Spam Detector API")

@app.post("/predict", response_model=PredictionAndExplanation)
def predict(request: PredictionRequest):
    prediction_and_explanation = predict_and_explain(request.message)
    return prediction_and_explanation