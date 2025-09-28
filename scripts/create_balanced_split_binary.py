#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_CSV = os.path.join(PROCESSED_PATH, "kdd_combined_processed.csv")
TRAIN_CSV = os.path.join(PROCESSED_PATH, "kdd_train_balanced.csv")
TEST_CSV = os.path.join(PROCESSED_PATH, "kdd_test_balanced.csv")
SPLIT_INFO_JSON = os.path.join(PROCESSED_PATH, "balanced_split_info.json")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------
# LOAD DATA
# -----------------------------
print("[INFO] Loading combined dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"[INFO] Dataset shape: {df.shape}")

# -----------------------------
# CONVERT TO BINARY LABEL
# -----------------------------
label_col = df.columns[42]
df["binary_label"] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

print("[INFO] Original binary label counts:")
print(df["binary_label"].value_counts())

# -----------------------------
# BALANCE DATA (oversample minority class)
# -----------------------------
normal_df = df[df["binary_label"] == 0]
attack_df = df[df["binary_label"] == 1]

max_size = max(len(normal_df), len(attack_df))  # size of majority class
print(f"[INFO] Oversampling to {max_size} samples per class...")

normal_oversampled = normal_df.sample(n=max_size, replace=True, random_state=RANDOM_STATE)
attack_oversampled = attack_df.sample(n=max_size, replace=True, random_state=RANDOM_STATE)

df_balanced = pd.concat([normal_oversampled, attack_oversampled]).sample(frac=1, random_state=RANDOM_STATE)

print("[INFO] Balanced binary label counts:")
print(df_balanced["binary_label"].value_counts())

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df_balanced.drop(columns=["binary_label"])
y = df_balanced["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------
# SAVE BALANCED CSVS
# -----------------------------
train_df = X_train.copy()
train_df["binary_label"] = y_train
train_df.to_csv(TRAIN_CSV, index=False)

test_df = X_test.copy()
test_df["binary_label"] = y_test
test_df.to_csv(TEST_CSV, index=False)

# Save split info
split_info = {
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "train_class_counts": y_train.value_counts().to_dict(),
    "test_class_counts": y_test.value_counts().to_dict()
}
with open(SPLIT_INFO_JSON, "w") as f:
    json.dump(split_info, f, indent=4)

print("\n✅ Balanced datasets saved (via oversampling):")
print(f"  - Training: {TRAIN_CSV}")
print(f"  - Testing: {TEST_CSV}")
print(f"  - Split info: {SPLIT_INFO_JSON}")
print("\n🎯 Ready for training with oversampled binary balanced data!")
