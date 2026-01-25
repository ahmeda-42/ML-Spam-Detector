from pathlib import Path
from urllib.request import urlopen
import zipfile
import io
import pandas as pd


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
DATASET_FILE = "SMSSpamCollection"
DATA_DIR = Path("data")
MODEL_PATH = Path("artifacts/model.joblib")


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    dataset_path = DATA_DIR / DATASET_FILE

    if dataset_path.exists():
        return dataset_path

    print(f"Downloading dataset to {dataset_path} ...")

    with urlopen(DATASET_URL, timeout=30) as response:
        content = response.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # Extract all, then find the file
        zf.extractall(DATA_DIR)

    if not dataset_path.exists():
        raise FileNotFoundError(f"{DATASET_FILE} not found in ZIP")

    return dataset_path


def load_dataset(dataset_path):
    # Load data into X and Y variables
    df = pd.read_csv(
        dataset_path,
        sep="\t", # tab-separated values
        header=None, # no header row
        names=["label", "text"],
        encoding="latin-1"  # "utf-8" can't read accent marks like é, ñ, etc.
    )
    X = df["text"]
    Y = df["label"].map({"ham": 0, "spam": 1}) # replaces "ham" with 0 and "spam" with 1
    if Y.isna().any(): # if any label is NaN, raise an error
        raise ValueError("Unexpected labels found in dataset.")
    return X, Y