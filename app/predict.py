import joblib
import numpy as np
from pathlib import Path

def predict(model, messages):
    preds = model.predict(messages) # 0 for ham, 1 for spam
    probs = model.predict_proba(messages) # 2 columns: [prob of ham, prob of spam]
    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        results.append({
            "message": msg,
            "prediction": "spam" if pred == 1 else "not_spam",
            "confidence": round(max(prob) * 100, 3),
        })
    return results

def predict_and_explain(model, messages, top_k: int = 5):
    preds = model.predict(messages) # 0 for ham, 1 for spam
    probs = model.predict_proba(messages) # 2 columns: [prob of ham, prob of spam]
    results = []
    for msg, pred, prob in zip(messages, preds, probs):
        results.append({
            "message": msg,
            "prediction": "spam" if pred == 1 else "not_spam",
            "confidence": round(max(prob) * 100, 3),
            "explanation": explain(model, msg, top_k),
        })
    return results

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
            "direction": "increased spam likelihood" if contributions[i] > 0 else "decreased spam likelihood",
            "percent": round((abs(contributions[i]) / total_contribution) * 100, 2),
        })
    return explanation