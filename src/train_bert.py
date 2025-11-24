import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_transformer_model():
    print("Loading data for Transformer Model...")
    df = pd.read_csv("data/imdb.csv")
    
    # Using a subset for demonstration speed
    train_df = df[df['split'] == 'train'].sample(2000, random_state=42) 
    test_df = df[df['split'] == 'test'].sample(500, random_state=42)
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="models/distilbert_checkpoints",
        eval_strategy="epoch",      # <--- FIX: Changed from evaluation_strategy
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1, 
        weight_decay=0.01,
        logging_steps=50,
        use_cpu=False if torch.cuda.is_available() else True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving Transformer model...")
    model.save_pretrained("models/saved_model/distilbert_sentiment")
    tokenizer.save_pretrained("models/saved_model/distilbert_sentiment")
    print("Transformer model saved.")

if __name__ == "__main__":
    train_transformer_model()