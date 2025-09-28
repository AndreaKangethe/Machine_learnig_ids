#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_balanced_simple():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # Load data
    train_df = pd.read_csv(os.path.join(raw_path, "KDDTrain+.txt"), header=None)
    test_df = pd.read_csv(os.path.join(raw_path, "KDDTest+.txt"), header=None)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Combine datasets
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Get labels (last column)
    labels = combined.iloc[:, -1]
    features = combined.iloc[:, :-1]
    
    print(f"Combined shape: {combined.shape}")
    print("Label distribution:")
    label_counts = labels.value_counts()
    for label, count in label_counts.head(10).items():
        pct = (count / len(labels)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Based on analysis, try label 18 as normal (16.4% is reasonable)
    NORMAL_LABEL = 18
    
    # Create binary labels: 0 = normal, 1 = attack
    binary_labels = (labels != NORMAL_LABEL).astype(int)
    
    print(f"\nUsing label {NORMAL_LABEL} as normal:")
    normal_count = (binary_labels == 0).sum()
    attack_count = (binary_labels == 1).sum()
    print(f"Normal: {normal_count} ({normal_count/len(binary_labels)*100:.1f}%)")
    print(f"Attack: {attack_count} ({attack_count/len(binary_labels)*100:.1f}%)")
    
    # Check if we have both classes
    if len(binary_labels.unique()) < 2:
        print("ERROR: Still only one class!")
        return False
    
    # Encode categorical features (columns 1, 2, 3 are typically categorical in KDD)
    X = features.copy()
    categorical_cols = [1, 2, 3]  # protocol_type, service, flag
    
    for col in categorical_cols:
        if X.iloc[:, col].dtype == "object":
            X.iloc[:, col] = X.iloc[:, col].astype("category").cat.codes
    
    # Create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels
    )
    
    print(f"\nBalanced split created:")
    print(f"Train: {X_train.shape}, Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
    print(f"Test:  {X_test.shape}, Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    
    # Create final dataframes
    train_final = X_train.copy()
    train_final['label'] = y_train
    
    test_final = X_test.copy() 
    test_final['label'] = y_test
    
    # Save
    os.makedirs(processed_path, exist_ok=True)
    train_file = os.path.join(processed_path, "kdd_train_balanced.csv")
    test_file = os.path.join(processed_path, "kdd_test_balanced.csv")
    
    train_final.to_csv(train_file, index=False)
    test_final.to_csv(test_file, index=False)
    
    print(f"\nSUCCESS: Saved balanced datasets:")
    print(f"  - {train_file}")  
    print(f"  - {test_file}")
    
    return True

if __name__ == "__main__":
    create_balanced_simple()
