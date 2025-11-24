import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_classical_model():
    print("Loading data for Classical Model...")
    df = pd.read_csv("data/imdb.csv")
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test'] # Used for quick validation
    
    # Vectorization
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train = tfidf.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    # Model Training
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Quick check
    X_test = tfidf.transform(test_df['text'])
    acc = accuracy_score(test_df['label'], model.predict(X_test))
    print(f"Classical Model Training Accuracy: {acc:.4f}")
    
    # Save artifacts
    os.makedirs("models/saved_model", exist_ok=True)
    joblib.dump(tfidf, "models/saved_model/tfidf_vectorizer.pkl")
    joblib.dump(model, "models/saved_model/logreg_model.pkl")
    print("Classical model saved to models/saved_model/")

if __name__ == "__main__":
    train_classical_model()