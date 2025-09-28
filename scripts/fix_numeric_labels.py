#!/usr/bin/env python3
# scripts/fix_numeric_labels.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

def analyze_numeric_labels():
    """Analyze the numeric labels to understand what they represent."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    
    kdd_train_file = os.path.join(raw_path, "KDDTrain+.txt")
    
    print("=== ANALYZING NUMERIC LABELS ===\n")
    
    # Read the training data
    df = pd.read_csv(kdd_train_file, header=None)
    labels = df.iloc[:, -1]  # Last column
    
    print(f"Total samples: {len(labels)}")
    print(f"Unique labels: {sorted(labels.unique())}")
    print(f"Number of unique labels: {len(labels.unique())}")
    
    # Show distribution
    print(f"\nLabel distribution:")
    label_counts = labels.value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(labels)) * 100
        print(f"  {label}: {count:6d} ({percentage:5.1f}%)")
    
    # KDD Cup 99 typically has:
    # - 1 normal class
    # - 4 main attack categories: DoS, Probe, R2L, U2R
    # - About 23 specific attack types
    
    # The normal class is usually the most frequent or has a specific pattern
    # Let's make some educated guesses:
    
    print(f"\n=== ANALYSIS ===")
    
    # Check if any label dominates (could be normal)
    max_count_label = label_counts.idxmax()
    max_count = label_counts.max()
    max_percentage = (max_count / len(labels)) * 100
    
    print(f"Most frequent label: {max_count_label} with {max_count} samples ({max_percentage:.1f}%)")
    
    # In original KDD, normal traffic is usually 15-20% of training data
    # But this could be encoded differently
    
    # Let's try different assumptions:
    assumptions = [
        (0, "Label 0 = Normal (common encoding)"),
        (1, "Label 1 = Normal (common encoding)"),
        (max_count_label, f"Most frequent label ({max_count_label}) = Normal"),
        (min(labels), f"Smallest label ({min(labels)}) = Normal"),
        (max(labels), f"Largest label ({max(labels)}) = Normal")
    ]
    
    print(f"\n=== TESTING DIFFERENT ASSUMPTIONS ===")
    
    for normal_label, description in assumptions:
        if normal_label in labels.values:
            normal_count = (labels == normal_label).sum()
            attack_count = len(labels) - normal_count
            normal_pct = (normal_count / len(labels)) * 100
            
            print(f"\n{description}:")
            print(f"  Normal samples: {normal_count:6d} ({normal_pct:5.1f}%)")
            print(f"  Attack samples: {attack_count:6d} ({100-normal_pct:5.1f}%)")
            
            # Check if this makes sense (normal should be 10-30% typically)
            if 5 <= normal_pct <= 40:
                print(f"  ✅ Reasonable distribution for intrusion detection")
            else:
                print(f"  ❓ Unusual distribution - might not be correct")
    
    return label_counts

def create_binary_labels_with_assumption(normal_label_value):
    """Create balanced dataset with specific normal label assumption."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # Load raw data
    train_file = os.path.join(raw_path, "KDDTrain+.txt")
    test_file = os.path.join(raw_path, "KDDTest+.txt")
    
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    
    print(f"\n=== CREATING BINARY LABELS (Normal = {normal_label_value}) ===")
    
    # Add proper feature names
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
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'original_label'
    ]
    
    train_df.columns = feature_names
    test_df.columns = feature_names
    
    # Create binary labels
    train_df['binary_label'] = (train_df['original_label'] == normal_label_value).astype(int)
    test_df['binary_label'] = (test_df['original_label'] == normal_label_value).astype(int)
    
    # Note: 1 = normal, 0 = attack (opposite of usual for this function)
    # Let's fix this to standard: 0 = normal, 1 = attack
    train_df['binary_label'] = 1 - train_df['binary_label']
    test_df['binary_label'] = 1 - test_df['binary_label']
    
    # Check distributions
    print(f"Training set:")
    train_counts = train_df['binary_label'].value_counts().sort_index()
    for label, count in train_counts.items():
        label_name = "Normal" if label == 0 else "Attack"
        pct = (count / len(train_df)) * 100
        print(f"  {label_name}: {count:6d} ({pct:5.1f}%)")
    
    print(f"Test set:")
    test_counts = test_df['binary_label'].value_counts().sort_index()
    for label, count in test_counts.items():
        label_name = "Normal" if label == 0 else "Attack"
        pct = (count / len(test_df)) * 100
        print(f"  {label_name}: {count:6d} ({pct:5.1f}%)")
    
    # Combine and create balanced split
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Prepare features (encode categorical)
    X = combined_df.drop(columns=['original_label', 'binary_label'])
    y = combined_df['binary_label']
    
    # Encode categorical columns
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == "object":
            categorical_cols.append(col)
            X[col] = X[col].astype("category").cat.codes
    
    print(f"Encoded categorical columns: {categorical_cols}")
    
    # Create stratified split
    if len(y.unique()) == 2:  # Only if we have both classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create final datasets
        train_final = X_train.copy()
        train_final['label'] = y_train
        
        test_final = X_test.copy()
        test_final['label'] = y_test
        
        # Save
        os.makedirs(processed_path, exist_ok=True)
        train_balanced_file = os.path.join(processed_path, f"kdd_train_balanced_normal_{normal_label_value}.csv")
        test_balanced_file = os.path.join(processed_path, f"kdd_test_balanced_normal_{normal_label_value}.csv")
        
        train_final.to_csv(train_balanced_file, index=False)
        test_final.to_csv(test_balanced_file, index=False)
        
        print(f"\n✅ Balanced datasets created:")
        print(f"  - Training: {train_balanced_file}")
        print(f"  - Testing: {test_balanced_file}")
        
        # Final verification
        print(f"\nFinal balanced sets:")
        print(f"Training: Normal={sum(y_train == 0)}, Attack={sum(y_train == 1)}")
        print(f"Test: Normal={sum(y_test == 0)}, Attack={sum(y_test == 1)}")
        
        return True
    else:
        print(f"❌ Still only one class with assumption normal={normal_label_value}")
        return False

if __name__ == "__main__":
    # First analyze what we have
    label_counts = analyze_numeric_labels()
    
    print(f"\n" + "="*50)
    print("RECOMMENDATION:")
    print("="*50)
    
    # Make a smart guess based on typical KDD patterns
    # Usually the normal class has a reasonable percentage (10-30%)
    total_samples = label_counts.sum()
    
    candidates = []
    for label, count in label_counts.items():
        pct = (count / total_samples) * 100
        if 5 <= pct <= 40:  # Reasonable range for normal traffic
            candidates.append((label, count, pct))
    
    if candidates:
        print(f"Most likely candidates for 'normal' label:")
        for label, count, pct in sorted(candidates, key=lambda x: abs(x[2] - 20)):  # Sort by closeness to 20%
            print(f"  Label {label}: {count} samples ({pct:.1f}%) - {'✅ BEST GUESS' if label == sorted(candidates, key=lambda x: abs(x[2] - 20))[0][0] else '❓ Possible'}")
        
        # Try the best guess
        best_guess = sorted(candidates, key=lambda x: abs(x[2] - 20))[0][0]
        print(f"\n🎯 Trying with normal_label = {best_guess}...")
        success = create_binary_labels_with_assumption(best_guess)
        
        if success:
            print(f"🎉 SUCCESS! Use normal_label = {best_guess}")
        else:
            print(f"❌ That didn't work. Try manually with other candidates.")
    else:
        print("❌ No clear candidate for normal label found.")
        print("This might be a dataset with only attack samples, or unusual encoding.")
