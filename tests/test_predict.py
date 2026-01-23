from pathlib import Path
import pytest
from app.model import load_model
from app.predict import predict, predict_and_explain


MODEL_PATH = Path("artifacts/model.joblib")

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_predict_returns_expected_fields_ham():
    model = load_model()
    result = predict(model, "hey are you coming later")

    assert result["message"] == "hey are you coming later"
    assert result["prediction"] == "not_spam"
    assert 0 <= result["confidence"] <= 100

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_predict_returns_expected_fields_spam():
    model = load_model()
    result = predict(model, "win free money now")

    assert result["message"] == "win free money now"
    assert result["prediction"] == "spam"
    assert 0 <= result["confidence"] <= 100


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact missing.")
def test_predict_and_explain_returns_explanation():
    model = load_model()
    result = predict_and_explain(model, "win free money now", top_k=3)

    assert result["prediction"] == "spam"
    assert 0 <= result["confidence"] <= 100
    assert isinstance(result["explanation"], list)
    if result["explanation"]:
        item = result["explanation"][0]
        assert set(item.keys()) == {"word", "direction", "percent"}
