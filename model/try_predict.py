from pathlib import Path
import joblib
from predict import predict, predict_and_explain

MODEL_PATH = Path("artifacts/model.joblib")
MESSAGES = [
    "Win free money now",
    "Hey are you coming later?",
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
predict_and_explain(model, MESSAGES)