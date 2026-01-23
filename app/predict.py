import joblib
import numpy as np
from pathlib import Path

def predict(model, message):
    preds = model.predict([message]) # 0 for ham, 1 for spam
    probs = model.predict_proba([message]) # 2 columns: [prob of ham, prob of spam]
    return {
        "message": message,
        "prediction": "spam" if preds[0] == 1 else "not_spam",
        "confidence": round(max(probs[0]) * 100, 3),
    }

def predict_and_explain(model, message, top_k: int = 5):
    preds = model.predict([message]) # 0 for ham, 1 for spam
    probs = model.predict_proba([message]) # 2 columns: [prob of ham, prob of spam]
    return {
        "message": message,
        "prediction": "spam" if preds[0] == 1 else "not_spam",
        "confidence": round(max(probs[0]) * 100, 3),
        "explanation": explain(model, message, top_k),
    }

def explain(model, msg, top_k: int = 5):
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
    
    total_contribution = np.sum(np.abs(contributions[sorted_indices])) or 1.0
    
    print("\nWhy? (Percentages show relative contributions of words to the model's decision, not absolute probability)")
    explanation = []
    for i in sorted_indices[:top_k]:
        explanation.append({
            "word": feature_names[i],
            "direction": "spam" if contributions[i] > 0 else "not_spam",
            "percent": round((abs(contributions[i]) / total_contribution) * 100, 2),
        })
    return explanation