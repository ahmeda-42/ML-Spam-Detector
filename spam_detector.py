#!/usr/bin/env python3
import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
DATASET_FILE = "SMSSpamCollection"


def download_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / DATASET_FILE
    if dataset_path.exists():
        return dataset_path

    print(f"Downloading dataset to {dataset_path} ...")
    with urlopen(DATASET_URL, timeout=30) as response:
        content = response.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extract(DATASET_FILE, path=data_dir)

    return dataset_path

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    if df["label"].isna().any():
        raise ValueError("Unexpected labels found in dataset.")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(
    dataset_path: Path, model_path: Path, test_size: float, random_state: int
) -> None:
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")


def predict_messages(model_path: Path, messages: list[str]) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline: Pipeline = joblib.load(model_path)
    preds = pipeline.predict(messages)
    for message, pred in zip(messages, preds, strict=False):
        label = "spam" if pred == 1 else "ham"
        print(f"[{label}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and use an ML spam detector (SMS Spam Collection)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a spam detector")
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store/download the dataset",
    )
    train_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/model.joblib"),
        help="Where to save the trained model",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict on messages")
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/model.joblib"),
        help="Path to a trained model",
    )
    predict_parser.add_argument(
        "messages",
        nargs="+",
        help="One or more messages to classify",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        dataset_path = download_dataset(args.data_dir)
        train_model(
            dataset_path=dataset_path,
            model_path=args.model_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    elif args.command == "predict":
        predict_messages(args.model_path, args.messages)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
