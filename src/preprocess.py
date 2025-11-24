import os
import pandas as pd
from datasets import load_dataset

def prepare_data():
    print("Downloading IMDB dataset (this may take a moment)...")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("imdb")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Convert to pandas DataFrames
    print("Converting to DataFrame...")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Add a column to distinguish splits
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Combine
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Validation check
    if full_df.empty:
        print("ERROR: The downloaded dataframe is empty!")
        return
    else:
        print(f"Success! Data loaded. Total rows: {len(full_df)}")

    # --- PATH FIX: Always find the project root ---
    # Get the directory where this script (preprocess.py) lives
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Go up one level to find the 'LLMAAT' root
    project_root = os.path.dirname(script_dir)
    # Define the data path relative to root
    output_dir = os.path.join(project_root, "data")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "imdb.csv")
    
    print(f"Saving dataset to: {output_path}")
    full_df.to_csv(output_path, index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()