from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from model.dataset import download_dataset, load_dataset
from model.load_model import load_model

def main():
    # Download dataset and get the path
    dataset_path = download_dataset()

    # Load data into X and Y variables
    X, Y = load_dataset(dataset_path)

    # randomly split data into training and testing sets
    _, X_test, _, Y_test = train_test_split(
        X,
        Y,
        test_size=0.05,  # 5% of data for testing
        random_state=67,  # random seed (for reproducibility)
        stratify=Y,
    )

    # Load model
    model = load_model()

    # Evaluate/Test model on testing set
    Y_pred = model.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(Y_test, Y_pred))
    print("\nClassification report:")
    print(classification_report(Y_test, Y_pred, digits=4))


if __name__ == "__main__":
    main()
