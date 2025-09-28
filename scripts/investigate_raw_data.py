#!/usr/bin/env python3
# scripts/investigate_raw_data.py
import pandas as pd
import os

def investigate_raw_data():
    """Investigate what's in the original raw KDD files."""
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    
    kdd_train_file = os.path.join(raw_path, "KDDTrain+.txt")
    kdd_test_file = os.path.join(raw_path, "KDDTest+.txt")
    
    print("=== INVESTIGATING RAW KDD DATA ===\n")
    
    # Check if files exist
    for name, file_path in [("Training", kdd_train_file), ("Test", kdd_test_file)]:
        print(f"[INFO] Checking {name} file: {file_path}")
        if os.path.exists(file_path):
            print(f"  ✅ File exists")
            
            # Read file and check basic info
            try:
                df = pd.read_csv(file_path, header=None)
                print(f"  📊 Shape: {df.shape}")
                
                # Check last column (should be labels)
                last_col = df.iloc[:, -1]
                print(f"  🏷️  Last column (labels) - unique values:")
                
                # Get sample of labels
                label_sample = last_col.head(20).tolist()
                print(f"    First 20 labels: {label_sample}")
                
                # Get unique labels and their counts
                unique_labels = last_col.value_counts()
                print(f"    Label distribution:")
                for label, count in unique_labels.head(10).items():  # Show top 10
                    print(f"      '{label}': {count}")
                
                if len(unique_labels) > 10:
                    print(f"    ... and {len(unique_labels) - 10} more label types")
                
                # Check for 'normal' specifically
                normal_count = sum(1 for x in last_col if str(x).strip().lower() == 'normal')
                total_count = len(last_col)
                attack_count = total_count - normal_count
                
                print(f"  🎯 Binary classification breakdown:")
                print(f"    Normal samples: {normal_count} ({normal_count/total_count*100:.1f}%)")
                print(f"    Attack samples: {attack_count} ({attack_count/total_count*100:.1f}%)")
                
                # Check some specific attack types
                attack_types = []
                for label in unique_labels.index[:10]:
                    if str(label).strip().lower() != 'normal':
                        attack_types.append(label)
                
                if attack_types:
                    print(f"    Sample attack types: {attack_types[:5]}")
                
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
                
        else:
            print(f"  ❌ File not found!")
        
        print()
    
    # Also check processed data to see what happened during preprocessing
    print("\n=== CHECKING PROCESSED DATA ===")
    processed_path = os.path.join(project_root, "data", "processed")
    
    processed_files = [
        "kdd_train_processed.csv",
        "kdd_test_processed.csv", 
        "kdd_combined_processed.csv"
    ]
    
    for filename in processed_files:
        file_path = os.path.join(processed_path, filename)
        if os.path.exists(file_path):
            print(f"[INFO] Checking {filename}:")
            try:
                df = pd.read_csv(file_path)
                print(f"  📊 Shape: {df.shape}")
                print(f"  🏷️  Columns: {list(df.columns)}")
                
                # Check if there's a label column
                label_cols = [col for col in df.columns if 'label' in col.lower()]
                if label_cols:
                    for label_col in label_cols:
                        print(f"  📊 {label_col} distribution:")
                        counts = df[label_col].value_counts()
                        for val, count in counts.head(10).items():
                            print(f"    {val}: {count}")
                else:
                    # Check last column
                    last_col = df.columns[-1]
                    print(f"  📊 Last column '{last_col}' distribution:")
                    counts = df[last_col].value_counts()
                    for val, count in counts.head(10).items():
                        print(f"    {val}: {count}")
                        
            except Exception as e:
                print(f"  ❌ Error reading {filename}: {e}")
            print()

if __name__ == "__main__":
    investigate_raw_data()
