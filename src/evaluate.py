import pandas as pd
import joblib
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_models():
    print("Loading Test Data...")
    df = pd.read_csv("data/imdb.csv")
    test_df = df[df['split'] == 'test'].sample(1000, random_state=42) # Subset for speed
    y_true = test_df['label'].values
    texts = test_df['text'].values.tolist()
    
    print("-" * 40)
    print("1. EVALUATING CLASSICAL MODEL (TF-IDF + LogReg)")
    try:
        tfidf = joblib.load("models/saved_model/tfidf_vectorizer.pkl")
        logreg = joblib.load("models/saved_model/logreg_model.pkl")
        
        X_test = tfidf.transform(texts)
        y_pred_classic = logreg.predict(X_test)
        
        print("\nClassical Model Report:")
        print(classification_report(y_true, y_pred_classic, target_names=['Negative', 'Positive']))
    except FileNotFoundError:
        print("Classical model artifacts not found. Run src/train_tfidf.py first.")

    print("-" * 40)
    print("2. EVALUATING TRANSFORMER MODEL (DistilBERT)")
    try:
        model_path = "models/saved_model/distilbert_sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Simple inference loop
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        y_pred_bert = torch.argmax(logits, dim=-1).numpy()
        
        print("\nTransformer Model Report:")
        print(classification_report(y_true, y_pred_bert, target_names=['Negative', 'Positive']))
        
    except Exception as e:
        print(f"Transformer model issue: {e}")
        print("Ensure you ran src/train_bert.py first.")

if __name__ == "__main__":
    evaluate_models()