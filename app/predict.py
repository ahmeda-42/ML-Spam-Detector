import joblib
import numpy as np
from pathlib import Path

def load_model():
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def explain(model, msg):
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    X = vectorizer.transform([msg])
    feature_names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    contributions = X.toarray()[0] * weights

    # Only keep words actually in the message
    nonzero_indices = np.where(X.toarray()[0] > 0)[0]

    # Sort by absolute contribution
    sorted_indices = nonzero_indices[np.argsort(np.abs(contributions[nonzero_indices]))[::-1]]
    
    total_contribution = np.sum(np.abs(contributions[sorted_indices]))
    
    for i in sorted_indices[:9]:
        word = feature_names[i]
        value = contributions[i]
        direction = "increased spam likelihood" if value > 0 else "decreased spam likelihood"
        percent = (abs(value) / total_contribution) * 100
        print(f'• "{word}" {direction} by {percent:.1f}%')


def predict(messages):
    model = load_model()
    preds = model.predict(messages) # 0 for ham, 1 for spam
    probs = model.predict_proba(messages) # 2 columns: [prob of ham, prob of spam]
    for msg, pred, prob in zip(messages, preds, probs):
        print("\nText:", msg)
        print("Prediction:", "SPAM" if pred == 1 else "NOT SPAM")
        print("Confidence:", max(prob)*100)
        print("\nWhy?")
        explain(model, msg)