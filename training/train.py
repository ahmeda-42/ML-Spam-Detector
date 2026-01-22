from pathlib import Path
from urllib.request import urlopen
import zipfile
import io
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
DATASET_FILE = "SMSSpamCollection"
DATA_DIR = Path("data")

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


# Download dataset and get the path
dataset_path = download_dataset()

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

# randomly split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.1, # 10% of data for testing
    random_state=67, # random seed (for reproducibility)
    stratify=Y, # ensures that the training and testing sets have the same distribution of labels
)

# Build pipeline model (logistic regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")), # lowercase and remove stop words (common words like "the", "and", "is", etc. to reduce noise)
    ("classifier", LogisticRegression(max_iter=1000)) 
])

# Train model
model.fit(X_train, Y_train)

# Evaluate/Test model
Y_pred = model.predict(X_test)
print("Confusion matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification report:")
print(classification_report(Y_test, Y_pred, digits=4))

# Save model
joblib.dump(model, "artifacts/model.joblib")

print("Model trained and saved to artifacts/model.joblib.")