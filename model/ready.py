import joblib
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("best_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

text = ["Free prize winner! Call now!"]
X = vectorizer.transform(text)
pred = model.predict(X)
print("Prediction:", le.inverse_transform(pred)[0])
