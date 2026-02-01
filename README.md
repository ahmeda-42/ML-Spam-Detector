# ML Spam Detector API with Explainability

A machine learning system that classifies/detects SMS spam messages, exposes predictions via a FastAPI JSON API, and explains why the model made each decision using interpretable feature contributions. You can interact with the API from simple CLI utilities. 

The model uses TF-IDF vectorization with logistic regression (scikit-learn). It is trained on the UCI SMS Spam Collection dataset and is saved as `artifacts/model.joblib`.

## Features

Spam classification using:
- TF-IDF vectorization
- Logistic Regression (scikit-learn)

Model explainability:
- Shows which words influenced the prediction
- Percent-based contribution for interpretability

FastAPI backend:
- /predict endpoint for real-time inference

Human-friendly CLI client:
- Pretty-printed predictions and explanations

Pytest test suite:
- Model tests
- Prediction logic tests
- API endpoint tests

Dockerized:
- One-command containerized deployment

## Example Output

### `utils/call_api.py` output

```md
API Response:
{
    "message": "free phone now",
    "prediction": "spam",
    "confidence": 52.515,
    "explanation": [
        {
            "word": "free",
            "direction": "spam",
            "percent": 79.99
        },
        {
            "word": "phone",
            "direction": "spam",
            "percent": 20.01
        }
    ]
}
```

### `utils/call_api_pretty.py` output

```md
"free phone now"

Prediction: SPAM
Confidence: 52.515%

Why? (Percentages show relative contributions of words to the model's decision, not absolute probability)
• "free" increased spam likelihood by 79.99%
• "phone" increased spam likelihood by 20.01%
```

## Project Structure

``` bash
ML-Spam-Detector-API/
├── app/                    # FastAPI application
│   ├── main.py             # API entry point with `/predict` endpoint
│   └── schemas.py          # Pydantic request/response models
├── model/                  # ML model training & evaluation
│   ├── dataset.py          # Dataset download + loading helpers
│   ├── train.py            # Trains/saves TF-IDF + Logistic Regression model
│   ├── evaluate.py         # Evaluates the saved model on the test split
│   ├── predict.py          # Prediction + explanation helpers
│   ├── load_model.py       # Helper function to load the model
│   └── try_predict.py      # Example batch predictions using the saved model
├── utils/                  # CLI utilities
│   ├── call_api.py         # Simple interactive API CLI client.
│   └── call_api_pretty.py  # Pretty-printed interactive API CLI client.
├── tests/                  # Pytest test suite for API, model loading, and predictions
│   ├── test_api.py
│   ├── test_model.py
│   └── test_predict.py
├── artifacts/              # Contains saved model
│   └── model.joblib
├── data/                   # Contains UCI SMS Spam Collection dataset
│   └── SMSSpamCollection
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech Stack

Machine Learning / Data Processing
- Pandas (data loading, preprocessing, exploration)
- scikit-learn (TF-IDF vectorization, Logistic Regression)

Backend
- FastAPI
- Pydantic
- Uvicorn

DevOps / SWE
- Docker
- Pytest
- Requests

## Try It Yourself!!

### 1. Setup virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### 2. Train the model

```bash
python model/train.py
```
This downloads the dataset to `data/SMSSpamCollection` (if missing), 
trains the model, and saves it to `artifacts/model.joblib`.


### 3. Evaluate the model

```bash
python model/evaluate.py
```
Evaluates on the held-out test split and prints the classification report and confusion matrix.


### 4. Run the API

```bash
uvicorn app.main:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Free entry in 2 a wkly comp"}'
```

Response fields:
- `message`: original input
- `prediction`: `spam` or `not_spam`
- `confidence`: percentage probability
- `explanation`: top contributing words and their impact

Note: Instead of running step 4 locally, you can run the service via the Docker section below.

### 5. Test the API from the CLI

After starting the API, in a seperate terminal run:

```bash
python utils/call_api.py
```

Or the pretty-printed variant:

```bash
python utils/call_api_pretty.py
```

Type messages directly into the terminal to see predictions and explanations.

Note: Example responses for both variants were shown above.

## Docker

Build the image (make sure `artifacts/model.joblib` exists first):

```bash
docker build -t ml-spam-detector .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 ml-spam-detector
```

## Testing

Run the full test suite:
```bash
pytest
```
Tests cover:
- Model loading
- Prediction correctness
- API endpoint behavior
