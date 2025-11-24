# LLMAAT: Sentiment Analysis of Movie Reviews

### Comparative Study: Classical ML vs. Transformer Models

## 📌 Project Overview
This project implements and compares two different Natural Language Processing (NLP) architectures for **Sentiment Analysis** on the IMDB Movie Reviews dataset. The goal is to classify reviews as either **Positive** or **Negative** and evaluate performance differences between traditional and modern approaches.

### 🚀 Architectures Implemented
1.  **Classical Approach:**
    * **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
    * **Model:** Logistic Regression
    * **Pros:** Fast training, interpretable, low resource usage.
2.  **Deep Learning Approach (SOTA):**
    * **Architecture:** DistilBERT (Transformer-based)
    * **Method:** Fine-tuning a pre-trained language model
    * **Pros:** Context-aware, handles complex sentence structures, State-of-the-Art accuracy.

---

## 📂 Project Structure

```text
LLMAAT/
│
├── data/               # Dataset storage
│   └── imdb.csv        # Processed IMDB dataset
│
├── models/             # Saved model artifacts
│   ├── saved_model/    # Local storage for pickles and pytorch binaries
│   └── distilbert.../  # Checkpoints during training
│
├── notebooks/
│   └── experiments.ipynb  # Visualization & Interactive Testing (Jupyter)
│
├── src/                # Source Code
│   ├── preprocess.py   # Downloads and prepares data
│   ├── train_tfidf.py  # Trains the Classical Model
│   ├── train_bert.py   # Trains the Transformer Model
│   └── evaluate.py     # Generates comparison metrics
│
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🛠️ Installation & Setup

1.  **Clone the repository (or extract the folder):**
    ```bash
    cd LLMAAT
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚡ Usage Guide

Run the pipeline steps in order using the terminal:

### Step 1: Data Preparation
Downloads the IMDB dataset from Hugging Face and converts it to a clean CSV format.
```bash
python src/preprocess.py
```

### Step 2: Train Classical Model (TF-IDF)
Vectorizes text and trains a Logistic Regression classifier. Saves artifacts to `models/saved_model/`.
```bash
python src/train_tfidf.py
```

### Step 3: Train Transformer Model (DistilBERT)
Fine-tunes the DistilBERT model.
*Note: This step requires significant RAM/GPU. If running on CPU, it may take time.*
```bash
python src/train_bert.py
```

### Step 4: Evaluate & Compare
Loads both models and prints a comparative classification report (Precision, Recall, F1-Score).
```bash
python src/evaluate.py
```

---

## 📊 Visualizations & Lab Notebook

Open `notebooks/experiments.ipynb` in VS Code or Jupyter Lab to see:
* **Data Distribution Plots** (Review lengths, Class balance)
* **Word Clouds** for Positive vs. Negative reviews
* **Confusion Matrices** comparing both models
* **Interactive Widget** to test your own custom movie reviews

---

## 📈 Results Summary

| Metric | Classical (TF-IDF) | Transformer (DistilBERT) |
| :--- | :--- | :--- |
| **Accuracy** | ~89% | **~93%** |
| **F1-Score** | 0.89 | **0.93** |
| **Training Time** | < 1 min | ~15-45 mins (CPU) |

**Conclusion:** While the TF-IDF model offers an excellent baseline with very fast training, the DistilBERT model outperforms it by capturing semantic context, proving superior for complex sentiment tasks.

---

## 📜 Requirements
* Python 3.8+
* Libraries: `pandas`, `scikit-learn`, `transformers`, `torch`, `datasets`, `matplotlib`, `seaborn`

---

**Authors:** Harsha Swaroop, Diksha CR

**Course:** NLP for LLM
