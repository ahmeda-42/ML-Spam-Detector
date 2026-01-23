from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app


MODEL_PATH = Path("artifacts/model.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_predict_endpoint():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"message": "free prize winner"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "explanation" in data

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_predict_endpoint_returns_prediction_and_explanation():
    client = TestClient(app)
    payload = {"message": "Win free money now!"}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["message"] == payload["message"]
    assert body["prediction"] in {"spam", "not_spam"}
    assert 0 <= body["confidence"] <= 100
    assert isinstance(body["explanation"], list)
    if body["explanation"]:
        item = body["explanation"][0]
        assert set(item.keys()) == {"word", "direction", "percent"}
        assert item["direction"] in {"spam", "not_spam"}
