# scripts/create_balanced_split.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

# ----------------------------
# Define processed data path
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

def create_balanced_split():
    """Create a proper train/test split with both classes represented."""
    
    # Load the combined processed data
    combined_file = os.path.join(PROCESSED_PATH, "kdd_combined_processed.csv")
    
    if not os.path.exists(combined_file):
        print(f"❌ Combined file not found: {combined_file}")
        print("Please run preprocess.py first")
        return
    
    print("[INFO] Loading combined dataset...")
    df = pd.read_csv(combined_file)
    print(f"[INFO] Combined dataset shape: {df.shape}")
    
    # Determine label column
    label_col = None
    potential_labels = ['label', 'binary_label', df.columns[-2]]
    for col in potential_labels:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("❌ No label column found in the dataset")
        return
    
    print(f"[INFO] Using label column: {label_col}")
    
    # Remove dataset indicator if exists
    feature_cols = [col for col in df.columns if col not in [label_col, 'dataset']]
    X = df[feature_cols]
    y = df[label_col]
    
    # Convert to binary if needed
    if not set(y.unique()).issubset({0, 1}):
        print("[INFO] Converting labels to binary (normal=0, attack=1)")
        y = y.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Save train/test splits
    train_df = X_train.copy()
    train_df['label'] = y_train
    test_df = X_test.copy()
    test_df['label'] = y_test
    
    train_file = os.path.join(PROCESSED_PATH, "kdd_train_balanced.csv")
    test_file = os.path.join(PROCESSED_PATH, "kdd_test_balanced.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✅ Balanced datasets saved:")
    print(f"  - Training: {train_file}")
    print(f"  - Testing: {test_file}")
    
    # Save split info
    split_info = {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_normal': int((y_train == 0).sum()),
        'train_attack': int((y_train == 1).sum()),
        'test_normal': int((y_test == 0).sum()),
        'test_attack': int((y_test == 1).sum()),
        'test_size': 0.2,
        'random_state': 42,
        'stratified': True
    }
    
    split_info_file = os.path.join(PROCESSED_PATH, "balanced_split_info.json")
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"  - Split info: {split_info_file}")
    print("\n🎯 Ready for training with balanced data!")

if __name__ == "__main__":
    create_balanced_split()
