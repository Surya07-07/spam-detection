# model/model_utils.py

import joblib

# Load saved model and vectorizer
model = joblib.load("model/best_model.pkl")
vectorizer = joblib.load("model/best_vectorizer.pkl")
le = joblib.load("model/label_encoder.pkl")

def predict(text):
    """
    text: Single string to classify
    returns: dict with prediction
    """
    # Wrap single string in a list
    X = vectorizer.transform([text])
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]  # get the first element
    return {"text": text, "prediction": label}
