#!/usr/bin/env python3
import os
import joblib
import pandas as pd

def migrate_model():
    # Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_dir = os.path.join(project_root, "src", "saved_models")
    data_dir = os.path.join(project_root, "data", "processed")

    old_model_path = os.path.join(model_dir, "rf_balanced_model.pkl")
    new_model_path = os.path.join(model_dir, "rf_balanced_model_fixed.pkl")
    train_file = os.path.join(data_dir, "kdd_train_balanced.csv")

    # Check files
    if not os.path.exists(old_model_path):
        print(f"❌ Old model file not found: {old_model_path}")
        return
    if not os.path.exists(train_file):
        print(f"❌ Training dataset not found: {train_file}")
        return

    print("🔄 Loading old model...")
    clf = joblib.load(old_model_path)

    # Recover feature names from training CSV
    print("📂 Loading training dataset to extract feature names...")
    train_df = pd.read_csv(train_file)
    features = list(train_df.drop(columns=['label']).columns)

    # Save in new format
    print("💾 Saving migrated model...")
    joblib.dump({
        "model": clf,
        "features": features
    }, new_model_path)

    print(f"✅ Migration complete! New model saved to:\n   {new_model_path}")
    print("\nYou can now update your app.py to load this new file instead.")

if __name__ == "__main__":
    migrate_model()
