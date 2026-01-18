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

## Notes

- The dataset is downloaded from UCI the first time you train.
- The model is saved with `joblib` and can be reused for predictions.
