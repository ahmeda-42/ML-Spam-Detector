from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from model.dataset import download_dataset, load_dataset

MODEL_OUT = "artifacts/model.joblib"

def main():
    # Download dataset and get the path
    dataset_path = download_dataset()

    # Load data into X and Y variables
    X, Y = load_dataset(dataset_path)

    # randomly split data into training and testing sets
    X_train, _, Y_train, _ = train_test_split(
        X,
        Y,
        test_size=0.05, # 5% of data for testing
        random_state=67, # random seed (for reproducibility)
        stratify=Y, # ensures that the training and testing sets have the same distribution of labels
    )

    # Build pipeline model (logistic regression)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")), # lowercase and remove stop words (common words like "the", "and", "is", etc. to reduce noise)
        ("classifier", LogisticRegression(max_iter=1000)) 
    ])

    # Train model on training set
    model.fit(X_train, Y_train)

    # Save model
    joblib.dump(model, MODEL_OUT)

    print(f"Model trained and saved to {MODEL_OUT}.")

if __name__ == "__main__":
    main()