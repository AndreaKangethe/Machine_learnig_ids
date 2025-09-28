# scripts/preprocess.py
import pandas as pd
import numpy as np
from mlids.config.config import KDD_TRAIN, KDD_TEST, PROCESSED_PATH
import os

def add_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add proper column names to KDD dataset."""
    # KDD Cup 99 feature names (41 features + 1 label)
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    if df.shape[1] == len(feature_names):
        df.columns = feature_names
    elif df.shape[1] == len(feature_names) - 1:
        # Missing label column
        df.columns = feature_names[:-1]
    else:
        print(f"[WARNING] Expected {len(feature_names)} or {len(feature_names)-1} columns, got {df.shape[1]}")
        print("[WARNING] Using default numeric column names")
    
    return df

def analyze_data_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Analyze and report data distribution."""
    print("\n=== DATA ANALYSIS ===")
    
    # Check if label column exists
    label_col = None
    for col in ['label', train_df.columns[-1]]:
        if col in train_df.columns:
            label_col = col
            break
    
    if label_col:
        print(f"\n[INFO] Training data label distribution:")
        train_labels = train_df[label_col].value_counts()
        print(train_labels)
        
        print(f"\n[INFO] Test data label distribution:")
        test_labels = test_df[label_col].value_counts()
        print(test_labels)
        
        # Check for normal vs attack distribution
        train_normal = sum(1 for x in train_df[label_col] if str(x).strip().lower() == 'normal')
        train_attack = len(train_df) - train_normal
        
        test_normal = sum(1 for x in test_df[label_col] if str(x).strip().lower() == 'normal')
        test_attack = len(test_df) - test_normal
        
        print(f"\n[INFO] Binary distribution:")
        print(f"Training: Normal={train_normal}, Attack={train_attack}")
        print(f"Test: Normal={test_normal}, Attack={test_attack}")
        
        if test_normal == 0 or test_attack == 0:
            print("⚠️  WARNING: Test set appears to contain only one class!")
            print("This will cause issues in model evaluation.")
    else:
        print("[WARNING] No label column found for analysis")

def preprocess_kdd():
    """Preprocess KDD Cup 99 dataset."""
    print("[INFO] Loading training data...")
    
    if not os.path.exists(KDD_TRAIN):
        raise FileNotFoundError(f"Training file not found: {KDD_TRAIN}")
    if not os.path.exists(KDD_TEST):
        raise FileNotFoundError(f"Test file not found: {KDD_TEST}")
    
    train_df = pd.read_csv(KDD_TRAIN, header=None)
    print(f"[INFO] Train shape: {train_df.shape}")
    
    print("[INFO] Loading testing data...")
    test_df = pd.read_csv(KDD_TEST, header=None)
    print(f"[INFO] Test shape: {test_df.shape}")
    
    # Add proper column names
    train_df = add_column_names(train_df)
    test_df = add_column_names(test_df)
    
    # Analyze data before processing
    analyze_data_distribution(train_df, test_df)
    
    print("\n[INFO] Processing data...")
    
    # Create copies to avoid modifying originals
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Track categorical columns for reporting
    categorical_cols = []
    
    # Encode categorical columns (object/string) → numbers
    for col in train_processed.columns:
        if train_processed[col].dtype == "object":
            categorical_cols.append(col)
            # Create consistent encoding across train/test
            combined_values = pd.concat([train_processed[col], test_processed[col]]).unique()
            cat_mapping = {val: idx for idx, val in enumerate(sorted(combined_values))}
            
            train_processed[col] = train_processed[col].map(cat_mapping)
            test_processed[col] = test_processed[col].map(cat_mapping)
    
    print(f"[INFO] Encoded categorical columns: {categorical_cols}")
    
    # Handle missing values
    missing_train = train_processed.isnull().sum().sum()
    missing_test = test_processed.isnull().sum().sum()
    
    if missing_train > 0 or missing_test > 0:
        print(f"[INFO] Found {missing_train} missing values in train, {missing_test} in test")
        train_processed = train_processed.fillna(0)
        test_processed = test_processed.fillna(0)
    
    # Merge train/test for combined dataset (optional)
    combined_df = pd.concat([train_processed, test_processed], ignore_index=True)
    combined_df['dataset'] = ['train'] * len(train_processed) + ['test'] * len(test_processed)
    
    # Create output directory
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # Save processed datasets
    train_file = os.path.join(PROCESSED_PATH, "kdd_train_processed.csv")
    test_file = os.path.join(PROCESSED_PATH, "kdd_test_processed.csv")
    combined_file = os.path.join(PROCESSED_PATH, "kdd_combined_processed.csv")
    
    train_processed.to_csv(train_file, index=False)
    test_processed.to_csv(test_file, index=False)
    combined_df.to_csv(combined_file, index=False)
    
    print(f"\n[INFO] Saved processed datasets:")
    print(f"  - Training: {train_file}")
    print(f"  - Testing: {test_file}")
    print(f"  - Combined: {combined_file}")
    
    # Save encoding information
    encoding_info = {
        'categorical_columns': categorical_cols,
        'feature_names': list(train_processed.columns),
        'train_shape': train_processed.shape,
        'test_shape': test_processed.shape
    }
    
    import json
    info_file = os.path.join(PROCESSED_PATH, "preprocessing_info.json")
    with open(info_file, 'w') as f:
        json.dump(encoding_info, f, indent=2)
    
    print(f"  - Info: {info_file}")
    print("\n✅ Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_kdd()
