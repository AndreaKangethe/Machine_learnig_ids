#!/usr/bin/env python3
# mlids/scripts/train_fnn_gpu_cumulative_log.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------
# 1. Import project config
# -------------------------------
from mlids.config.config import PROCESSED_PATH, SAVED_MODELS_PATH, LOGS_PATH

# Paths
train_csv = os.path.join(PROCESSED_PATH, "kdd_train_balanced.csv")
test_csv = os.path.join(PROCESSED_PATH, "kdd_test_balanced.csv")
model_save_path = os.path.join(SAVED_MODELS_PATH, "kdd_fnn_model_gpu_cumulative.pth")

# Cumulative log file
cumulative_log_file = os.path.join(LOGS_PATH, "training_logs", "fnn_training_cumulative.csv")
os.makedirs(os.path.dirname(cumulative_log_file), exist_ok=True)

# -------------------------------
# 2. Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 3. Load Dataset & Print Label Distribution
# -------------------------------
print("[INFO] Loading balanced datasets...")
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError(f"Balanced datasets not found. Please run create_balanced_split_binary.py first.\n"
                          f"Expected files:\n  - {train_csv}\n  - {test_csv}")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

print(f"[INFO] Training data shape: {df_train.shape}")
print(f"[INFO] Test data shape: {df_test.shape}")

# Check data types and identify non-numeric columns
print("[INFO] Checking data types...")
non_numeric_cols = df_train.select_dtypes(include=['object']).columns.tolist()
if 'binary_label' in non_numeric_cols:
    non_numeric_cols.remove('binary_label')

if non_numeric_cols:
    print(f"[WARNING] Found non-numeric columns: {non_numeric_cols}")
    print("[INFO] Converting non-numeric columns to numeric...")
    
    for col in non_numeric_cols:
        print(f"  Processing column: {col}")
        print(f"    Unique values: {df_train[col].unique()[:10]}...")  # Show first 10 unique values
        
        # Try to convert to numeric, replacing errors with 0
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)

# Use correct column name: "binary_label"
X_train = df_train.drop("binary_label", axis=1).values.astype(np.float32)
y_train = df_train["binary_label"].values.astype(np.int32)
X_test = df_test.drop("binary_label", axis=1).values.astype(np.float32)
y_test = df_test["binary_label"].values.astype(np.int32)

print("[INFO] Train label distribution:")
train_counts = pd.Series(y_train).value_counts().sort_index()
print(f"  Normal (0): {train_counts.get(0, 0)}")
print(f"  Attack (1): {train_counts.get(1, 0)}")
train_balance_ratio = train_counts.get(0, 0) / train_counts.get(1, 0) if train_counts.get(1, 0) > 0 else 0
print(f"  Balance ratio (Normal/Attack): {train_balance_ratio:.4f}")

print("[INFO] Test label distribution:")
test_counts = pd.Series(y_test).value_counts().sort_index()
print(f"  Normal (0): {test_counts.get(0, 0)}")
print(f"  Attack (1): {test_counts.get(1, 0)}")
test_balance_ratio = test_counts.get(0, 0) / test_counts.get(1, 0) if test_counts.get(1, 0) > 0 else 0
print(f"  Balance ratio (Normal/Attack): {test_balance_ratio:.4f}")

# Check for severe imbalance
if train_balance_ratio < 0.1 or train_balance_ratio > 10:
    print(f"\033[93mWARNING: Severe class imbalance detected! Consider rebalancing your dataset.\033[0m")
    print(f"         Current ratio suggests the balanced split may not have worked correctly.")

# Check data quality
print(f"[INFO] Data quality check:")
print(f"  Training set - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
print(f"  Test set - Min: {X_test.min():.4f}, Max: {X_test.max():.4f}")
print(f"  Training set has NaN: {np.isnan(X_train).any()}")
print(f"  Test set has NaN: {np.isnan(X_test).any()}")

# Handle NaN values if present
if np.isnan(X_train).any() or np.isnan(X_test).any():
    print("[INFO] Replacing NaN values with 0...")
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

print(f"[INFO] Feature dimensions: {X_train.shape[1]}")

# -------------------------------
# 4. Convert to PyTorch Tensors & Datasets
# -------------------------------
print("[INFO] Converting to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"[INFO] Batch size: {batch_size}")
print(f"[INFO] Training batches: {len(train_loader)}")
print(f"[INFO] Test batches: {len(test_loader)}")

# -------------------------------
# 5. Define FNN Model
# -------------------------------
class KDD_FNN(nn.Module):
    def __init__(self, input_dim):
        super(KDD_FNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Remove sigmoid since we're using BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.network(x)

model = KDD_FNN(input_dim=X_train.shape[1]).to(device)
print(f"[INFO] Model created with {sum(p.numel() for p in model.parameters())} parameters")

# -------------------------------
# 6. Loss & Optimizer
# -------------------------------
# Calculate class weights for imbalanced dataset
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32).to(device)

print(f"[INFO] Class weights - Normal: {class_weights[0]:.4f}, Attack: {class_weights[1]:.4f}")
print(f"[INFO] Positive weight for BCE: {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# -------------------------------
# 7. Training Loop with Early Stopping
# -------------------------------
print("\n[INFO] Starting training...")
num_epochs = 50
patience = 7
best_loss = np.inf
counter = 0
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
    loss_history.append(epoch_loss)
    scheduler.step(epoch_loss)

    print(f"Epoch {epoch+1:2d}/{num_epochs}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Early Stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        # Save best model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"    ✅ New best model saved (loss: {best_loss:.6f})")
    else:
        counter += 1
        if counter >= patience:
            print(f"    🛑 Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
print(f"\n[INFO] Loading best model for evaluation...")
model.load_state_dict(torch.load(model_save_path))

# -------------------------------
# 8. Evaluation
# -------------------------------
print("[INFO] Evaluating model on test set...")
model.eval()
y_pred_logits = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred_logits.extend(outputs.cpu().numpy().flatten())
        y_true.extend(y_batch.numpy().flatten())

y_pred_logits = np.array(y_pred_logits)
y_pred_prob = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()  # Convert logits to probabilities
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = np.array(y_true).astype(int)

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1_score": f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": None,
    "epochs_trained": len(loss_history),
    "final_loss": float(best_loss),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Calculate ROC AUC if both classes present
if len(np.unique(y_true)) > 1:
    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob)
else:
    print("\033[93mWARNING: ROC AUC not defined (only one class present in test set).\033[0m")

print("\n" + "="*50)
print("           EVALUATION RESULTS")
print("="*50)
for k, v in metrics.items():
    if k not in ["timestamp", "epochs_trained", "final_loss"]:
        if v is not None:
            print(f"{k.upper():>12}: {v:.4f}")
        else:
            print(f"{k.upper():>12}: N/A")

print(f"\n{'EPOCHS':>12}: {metrics['epochs_trained']}")
print(f"{'FINAL LOSS':>12}: {metrics['final_loss']:.6f}")

# Confusion Matrix details
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nCONFUSION MATRIX:")
print(f"{'':>10} {'Predicted':>20}")
print(f"{'':>10} {'Normal':>10} {'Attack':>10}")
print(f"{'Normal':>10} {tn:>10d} {fp:>10d}")
print(f"{'Attack':>10} {fn:>10d} {tp:>10d}")

# Security-specific metrics
detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nSECURITY METRICS:")
print(f"{'Detection Rate':>15}: {detection_rate:.4f} (TP/(TP+FN))")
print(f"{'False Alarm Rate':>15}: {false_alarm_rate:.4f} (FP/(FP+TN))")
print(f"{'Specificity':>15}: {1-false_alarm_rate:.4f} (TN/(TN+FP))")

# -------------------------------
# 9. Append to Cumulative Log
# -------------------------------
print(f"\n[INFO] Updating cumulative training log...")
metrics_extended = {**metrics, 
                   'detection_rate': detection_rate,
                   'false_alarm_rate': false_alarm_rate,
                   'specificity': 1-false_alarm_rate}

df_new = pd.DataFrame([metrics_extended])
if os.path.exists(cumulative_log_file):
    df_cum = pd.read_csv(cumulative_log_file)
    df_cum = pd.concat([df_cum, df_new], ignore_index=True)
else:
    df_cum = df_new

df_cum.to_csv(cumulative_log_file, index=False)
print(f"✅ Cumulative log updated: {cumulative_log_file}")

# -------------------------------
# 10. Visualizations
# -------------------------------
print(f"\n[INFO] Generating visualizations...")

# Create plots directory
plots_dir = os.path.join(LOGS_PATH, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Confusion Matrix Plot
if len(np.unique(y_true)) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal", "Attack"], 
                yticklabels=["Normal", "Attack"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - KDD FNN Binary Classification')
    
    # Add performance metrics as text
    roc_auc_str = f'{metrics["roc_auc"]:.3f}' if metrics["roc_auc"] is not None else "N/A"
    plt.figtext(0.02, 0.02, 
                f'Accuracy: {metrics["accuracy"]:.3f} | '
                f'F1-Score: {metrics["f1_score"]:.3f} | '
                f'ROC-AUC: {roc_auc_str}',
                fontsize=10, ha='left')
    
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Confusion matrix saved: {cm_plot_path}")

# Training Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', linewidth=2, markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Convergence")
plt.grid(True, alpha=0.3)
plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best Loss: {best_loss:.6f}')
plt.legend()
plt.tight_layout()

loss_plot_path = os.path.join(plots_dir, f"training_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Training loss plot saved: {loss_plot_path}")

print("\n" + "="*50)
print("           TRAINING COMPLETED")
print("="*50)
print(f"✅ Model saved: {model_save_path}")
print(f"✅ Logs updated: {cumulative_log_file}")
print(f"✅ Plots saved: {plots_dir}")
print(f"🎯 Ready for deployment!")
