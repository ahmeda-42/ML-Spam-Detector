# ML Spam Detector (SMS Spam Collection)

This project trains a simple spam detector using the UCI SMS Spam Collection
dataset. It downloads the dataset automatically, trains a TF-IDF + Naive Bayes
model, and saves it for later predictions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python spam_detector.py train
```

You can customize where data and model artifacts are stored:

```bash
python spam_detector.py train --data-dir data --model-path models/spam_detector.joblib
```

## Predict new messages

```bash
python spam_detector.py predict "Free entry in 2 a wkly comp" "Hey, are we still on for lunch?"
```

## Docker

Build the image (make sure `artifacts/model.joblib` exists first):

```bash
docker build -t ml-spam-detector .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 ml-spam-detector
```

Test the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Free entry in 2 a wkly comp"}'
```

## Notes

- The dataset is downloaded from UCI the first time you train.
- The model is saved with `joblib` and can be reused for predictions.
