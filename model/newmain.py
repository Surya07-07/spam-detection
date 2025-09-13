import pandas as pd
import numpy as np
import re
import string
import nltk
import warnings
import joblib
warnings.filterwarnings('ignore')

# Download NLTK resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv(r"dataset.csv")

# Encode labels to numbers (for all models including XGBoost)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # ham=0, spam=1

print(f"Dataset shape: {df.shape}")
print(df.head())


# =============================
# 2. TEXT CLEANING
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)


# =============================
# 3. TOKENIZATION, STOPWORDS, STEMMING, LEMMATIZATION
# =============================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_tokens(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['clean_text'].apply(preprocess_tokens)


# =============================
# 4. FEATURE EXTRACTION
# =============================
vectorizers = {
    "CountVectorizer": CountVectorizer(),
    "TF-IDF": TfidfVectorizer()
}

X_train_list = {}
X_test_list = {}
y_train = None
y_test = None

for name, vectorizer in vectorizers.items():
    X_train, X_test, y_train, y_test = train_test_split(
        vectorizer.fit_transform(df['processed_text']),
        df['label'],
        test_size=0.5 if len(df) < 20 else 0.2,  # big test size if small dataset
        random_state=42
    )
    X_train_list[name] = X_train
    X_test_list[name] = X_test


# =============================
# 5. CLASSIFIERS & PARAM GRIDS
# =============================
models = {
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.5, 1.0]}),
    "Logistic Regression": (LogisticRegression(max_iter=500), {"C": [0.1, 1]}),
    "SVM": (LinearSVC(), {"C": [0.1, 1]}),
    "SGD": (SGDClassifier(), {"loss": ["hinge", "log_loss"], "alpha": [1e-4]}),
    "Extra Trees": (ExtraTreesClassifier(), {"n_estimators": [50]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50]}),
    "MLP": (MLPClassifier(max_iter=500), {"hidden_layer_sizes": [(50,)], "alpha": [0.0001]}),
    "XGBoost": (xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), {"n_estimators": [50], "max_depth": [3]})
}


# =============================
# 6. TRAINING & EVALUATION
# =============================
results = []
best_f1 = -1
best_model = None
best_vec_name = None

for vec_name in vectorizers.keys():
    print(f"\n--- Feature Set: {vec_name} ---")
    for model_name, (model, params) in models.items():
        print(f"\nTraining {model_name} with {vec_name} features...")
        
        try:
            grid = GridSearchCV(model, params, cv=2 if len(df) < 20 else 3, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train_list[vec_name], y_train)
            
            best_est = grid.best_estimator_
            y_pred = best_est.predict(X_test_list[vec_name])
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results.append({
                "Vectorizer": vec_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1
            })
            
            print(classification_report(y_test, y_pred, zero_division=0))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = best_est
                best_vec_name = vec_name
        
        except Exception as e:
            print(f"⚠️ Skipping {model_name} due to error: {e}")


# =============================
# 7. SAVE BEST MODEL
# =============================
if best_model:
    print(f"\n✅ Best Model: {best_model} ({best_vec_name}) with F1-score={best_f1:.4f}")
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizers[best_vec_name], "best_vectorizer.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("💾 Best model, vectorizer, and label encoder saved!")


# =============================
# 8. SHOW RESULTS
# =============================
results_df = pd.DataFrame(results)
print("\n--- Final Results ---")
print(results_df.sort_values(by="F1-score", ascending=False))
