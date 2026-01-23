# ML Spam Detector with Explainability (SMS Spam Collection)

A machine learning system that classifies/detects SMS spam messages, expose predictions in a JSON via a FastAPI service, and explains why the model made each decision using interpretable feature contributions. You can interact with the API from simple CLI utilities. 
The model uses logistic regression (scikit-learn) and is TF-IDF vectorizes. It is trained on the UCI SMS Spam Collection dataset and is saved as `artifacts/model.joblib`.

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

```bash
"free phone now"

Prediction: SPAM
Confidence: 52.515%

Why? (Percentages show relative contributions of words to 
the model's decision, not absolute probability)
• "free" increased spam likelihood by 79.99%
• "phone" increased spam likelihood by 20.01%
```

## Project structure/layout

``` bash
ML-Spam-Detector/
├── app/                    # FastAPI application
│   ├── main.py             # API entry point with `/predict` endpoint
│   ├── model.py            # Model loading logic
│   ├── predict.py          # Prediction + explanation helpers
│   └── schemas.py          # Pydantic request/response models
├── training/               # ML training & evaluation
│   ├── train.py            # Downloads dataset, trains/saves TF-IDF + Logistic Regression model
│   ├── evaluate.py         # Example batch predictions using the saved model
│   └── predict.py          # CLI prediction + explanation helpers
├── utils/                  # CLI utilities
│   ├── call_api.py         # Simple interactive API client.
│   └── call_api_pretty.py  # Pretty-printed interactive API client.
├── tests/                  # Pytest test suite for API, model loading, and predictions
│   ├── test_api.py
│   ├── test_model.py
│   └── test_predict.py
├── artifacts/              # Containes saved model
│   └── model.joblib
├── data/                   # Containes UCI SMS Spam Collection dataset
│   └── SMSSpamCollection
├── Dockerfile
├── requirements.txt
└── README.md
```


## Try it

### 1. Setup virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### 2. Train the model

```bash
python training/train.py
```
This downloads the dataset to `data/SMSSpamCollection` (if missing), trains the
model, prints metrics, and saves the artifact to `artifacts/model.joblib`.

### 3. Run the API

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
- `explanation`: top contributing words

### 4. Test the API from the CLI

After starting the API, in a seperate terminal run:

```bash
python utils/call_api.py
```

Or the pretty-printed variant:

```bash
python utils/call_api_pretty.py
```

## Running tests

```bash
pytest
```
Tests cover:
- Model loading
- Prediction correctness
- API endpoint behavior

## Docker

Build the image (make sure `artifacts/model.joblib` exists first):

```bash
docker build -t ml-spam-detector .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 ml-spam-detector
```

## Tech Stack

Machine Learning
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

Backend
- FastAPI
- Pydantic
- Uvicorn

DevOps / SWE
- Docker
- Pytest
- Requests
