from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    message: str

class ExplanationItem(BaseModel):
    word: str
    direction: str
    percent: float

class Prediction(BaseModel):
    message: str
    prediction: str
    confidence: float

class PredictionAndExplanation(BaseModel):
    message: str
    prediction: str
    confidence: float
    explanation: List[ExplanationItem]