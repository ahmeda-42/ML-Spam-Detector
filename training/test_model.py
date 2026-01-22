from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from app.predict import predict

MESSAGES = [
    "Win free money now",
    "Hey are you coming later?",
]

predict(MESSAGES)