#!/usr/bin/env python3
# mlids/scripts/train_autoencoder_anomaly.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# -------------------------------
# 1. Import project config
# -------------------------------
from mlids.config.config import PROCESSED_PATH, SAVED_MODELS_PATH, LOGS_PATH

# Paths
train_csv = os.path.join(PROCESSED_PATH, "kdd_train_balanced.csv")
test_csv = os.path.join(PROCESSED_PATH, "kdd_test_balanced.csv")
model_save_path = os.path.join(SAVED_MODELS_PATH, "kdd_autoencoder_anomaly.pth")
scaler_save_path = os.path.join(SAVED_MODELS_PATH, "autoencoder_scaler.pkl")

# Logs
log_file = os.path.join(LOGS_PATH, "training_logs", "autoencoder_training.csv")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

# -------------------------------
# 2. Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 3. Load and Prepare Data
# -------------------------------
print("[INFO] Loading balanced datasets...")
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError(f"Balanced datasets not found. Please run create_balanced_split_binary.py first.")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

print(f"[INFO] Training data shape: {df_train.shape}")
print(f"[INFO] Test data shape: {df_test.shape}")

# Handle non-numeric columns
non_numeric_cols = df_train.select_dtypes(include=['object']).columns.tolist()
if 'binary_label' in non_numeric_cols:
    non_numeric_cols.remove('binary_label')

if non_numeric_cols:
    print(f"[INFO] Converting non-numeric columns: {non_numeric_cols}")
    for col in non_numeric_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)

# Separate features and labels
X_train = df_train.drop("binary_label", axis=1).values.astype(np.float32)
y_train = df_train["binary_label"].values.astype(np.int32)
X_test = df_test.drop("binary_label", axis=1).values.astype(np.int32)
y_test = df_test["binary_label"].values.astype(np.int32)

# Replace NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"[INFO] Feature dimensions: {X_train.shape[1]}")

# -------------------------------
# 4. Prepare Normal Traffic Only for Training
# -------------------------------
print("[INFO] Filtering normal traffic for autoencoder training...")

# Extract only normal traffic (label = 0) for training the autoencoder
normal_train_idx = (y_train == 0)
normal_test_idx = (y_test == 0)
attack_test_idx = (y_test == 1)

X_normal_train = X_train[normal_train_idx]
X_normal_test = X_test[normal_test_idx]  # Normal traffic for validation
X_attack_test = X_test[attack_test_idx]  # Attack traffic for anomaly detection

print(f"[INFO] Normal training samples: {len(X_normal_train)}")
print(f"[INFO] Normal test samples: {len(X_normal_test)}")
print(f"[INFO] Attack test samples: {len(X_attack_test)}")

# -------------------------------
# 5. Feature Scaling (Important for Autoencoders)
# -------------------------------
print("[INFO] Applying feature scaling...")
scaler = StandardScaler()
X_normal_train_scaled = scaler.fit_transform(X_normal_train)
X_normal_test_scaled = scaler.transform(X_normal_test)
X_attack_test_scaled = scaler.transform(X_attack_test)

# Save scaler for inference
joblib.dump(scaler, scaler_save_path)
print(f"[INFO] Scaler saved to: {scaler_save_path}")

# -------------------------------
# 6. Convert to PyTorch Tensors
# -------------------------------
print("[INFO] Converting to PyTorch tensors...")
X_normal_train_tensor = torch.tensor(X_normal_train_scaled, dtype=torch.float32)
X_normal_test_tensor = torch.tensor(X_normal_test_scaled, dtype=torch.float32)
X_attack_test_tensor = torch.tensor(X_attack_test_scaled, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_normal_train_tensor, X_normal_train_tensor)  # Input = Target for autoencoder
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"[INFO] Batch size: {batch_size}")
print(f"[INFO] Training batches: {len(train_loader)}")

# -------------------------------
# 7. Define Autoencoder Architecture
# -------------------------------
class KDDAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(KDDAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Bottleneck layer (compressed representation)
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),  # Reconstruct original input
            nn.Tanh()  # Output in range [-1, 1] after scaling
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Initialize model
input_dim = X_train.shape[1]
model = KDDAutoencoder(input_dim).to(device)
print(f"[INFO] Autoencoder created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"[INFO] Architecture: {input_dim} -> 32 -> 16 -> 8 -> 16 -> 32 -> {input_dim}")

# -------------------------------
# 8. Loss and Optimizer
# -------------------------------
criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# -------------------------------
# 9. Training Loop
# -------------------------------
print("\n[INFO] Starting autoencoder training on normal traffic only...")
num_epochs = 100
patience = 10
best_loss = np.inf
counter = 0
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_data, _ in train_loader:  # Input = Target for autoencoder
        batch_data = batch_data.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(batch_data)
        loss = criterion(reconstructed, batch_data)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    epoch_loss /= num_batches
    loss_history.append(epoch_loss)
    scheduler.step(epoch_loss)
    
    print(f"Epoch {epoch+1:3d}/{num_epochs}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        counter += 1
        if counter >= patience:
            print(f"    🛑 Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
print(f"\n[INFO] Loading best model for evaluation...")
model.load_state_dict(torch.load(model_save_path))

# -------------------------------
# 10. Anomaly Detection Evaluation
# -------------------------------
print("[INFO] Evaluating anomaly detection performance...")
model.eval()

def compute_reconstruction_error(model, data_tensor):
    """Compute reconstruction error for anomaly detection."""
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        reconstructed = model(data_tensor)
        errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)  # MSE per sample
        return errors.cpu().numpy()

# Compute reconstruction errors
normal_errors = compute_reconstruction_error(model, X_normal_test_tensor)
attack_errors = compute_reconstruction_error(model, X_attack_test_tensor)

print(f"[INFO] Normal traffic reconstruction error - Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
print(f"[INFO] Attack traffic reconstruction error - Mean: {np.mean(attack_errors):.6f}, Std: {np.std(attack_errors):.6f}")

# -------------------------------
# 11. Set Anomaly Threshold
# -------------------------------
# Use statistical approach: mean + k*std of normal reconstruction errors
k = 2.5  # Number of standard deviations
threshold = np.mean(normal_errors) + k * np.std(normal_errors)

print(f"[INFO] Anomaly detection threshold: {threshold:.6f}")
print(f"[INFO] Threshold = mean(normal_errors) + {k} * std(normal_errors)")

# Classify samples based on reconstruction error
normal_predictions = (normal_errors > threshold).astype(int)  # 1 = anomaly, 0 = normal
attack_predictions = (attack_errors > threshold).astype(int)

# Create ground truth labels (0 = normal, 1 = attack/anomaly)
y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(attack_errors))])
y_pred = np.concatenate([normal_predictions, attack_predictions])
y_scores = np.concatenate([normal_errors, attack_errors])  # Reconstruction errors as scores

# -------------------------------
# 12. Calculate Metrics
# -------------------------------
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_scores)

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

print("\n" + "="*60)
print("         AUTOENCODER ANOMALY DETECTION RESULTS")
print("="*60)
print(f"{'ACCURACY':>15}: {accuracy:.4f}")
print(f"{'ROC AUC':>15}: {roc_auc:.4f}")
print(f"{'PR AUC':>15}: {pr_auc:.4f}")
print(f"{'THRESHOLD':>15}: {threshold:.6f}")
print(f"{'EPOCHS TRAINED':>15}: {len(loss_history)}")
print(f"{'FINAL LOSS':>15}: {best_loss:.6f}")

# Classification report
print(f"\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nCONFUSION MATRIX:")
print(f"{'':>12} {'Predicted':>20}")
print(f"{'':>12} {'Normal':>10} {'Attack':>10}")
print(f"{'Normal':>12} {tn:>10d} {fp:>10d}")
print(f"{'Attack':>12} {fn:>10d} {tp:>10d}")

# Anomaly detection specific metrics
detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nANOMALY DETECTION METRICS:")
print(f"{'Detection Rate':>20}: {detection_rate:.4f} (TP/(TP+FN))")
print(f"{'False Alarm Rate':>20}: {false_alarm_rate:.4f} (FP/(FP+TN))")
print(f"{'Specificity':>20}: {1-false_alarm_rate:.4f} (TN/(TN+FP))")

# -------------------------------
# 13. Save Results and Metadata
# -------------------------------
print(f"\n[INFO] Saving model and metadata...")

# Save model metadata
metadata = {
    'input_dim': input_dim,
    'threshold': float(threshold),
    'threshold_method': f'mean + {k} * std',
    'architecture': f'{input_dim} -> 32 -> 16 -> 8 -> 16 -> 32 -> {input_dim}',
    'training_samples': len(X_normal_train),
    'epochs_trained': len(loss_history),
    'final_loss': float(best_loss),
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'pr_auc': float(pr_auc),
    'detection_rate': float(detection_rate),
    'false_alarm_rate': float(false_alarm_rate),
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

import json
metadata_path = os.path.join(SAVED_MODELS_PATH, "autoencoder_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Save training log
log_data = pd.DataFrame([metadata])
if os.path.exists(log_file):
    existing_log = pd.read_csv(log_file)
    log_data = pd.concat([existing_log, log_data], ignore_index=True)
log_data.to_csv(log_file, index=False)

# -------------------------------
# 14. Visualizations
# -------------------------------
print(f"[INFO] Generating visualizations...")
plots_dir = os.path.join(LOGS_PATH, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', linewidth=2, markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss (MSE)")
plt.title("Autoencoder Training Loss")
plt.grid(True, alpha=0.3)
plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best Loss: {best_loss:.6f}')
plt.legend()
plt.tight_layout()

loss_plot_path = os.path.join(plots_dir, f"autoencoder_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Reconstruction Error Distribution
plt.figure(figsize=(12, 6))
plt.hist(normal_errors, bins=50, alpha=0.7, label=f'Normal Traffic (n={len(normal_errors)})', color='blue', density=True)
plt.hist(attack_errors, bins=50, alpha=0.7, label=f'Attack Traffic (n={len(attack_errors)})', color='red', density=True)
plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

error_plot_path = os.path.join(plots_dir, f"reconstruction_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Autoencoder Anomaly Detection - Confusion Matrix')
plt.figtext(0.02, 0.02, 
            f'Accuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f} | Detection Rate: {detection_rate:.3f}',
            fontsize=10, ha='left')
plt.tight_layout()

cm_plot_path = os.path.join(plots_dir, f"autoencoder_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("         AUTOENCODER TRAINING COMPLETED")
print("="*60)
print(f"✅ Model saved: {model_save_path}")
print(f"✅ Scaler saved: {scaler_save_path}")
print(f"✅ Metadata saved: {metadata_path}")
print(f"✅ Logs updated: {log_file}")
print(f"✅ Plots saved: {plots_dir}")
print(f"🎯 Autoencoder ready for anomaly detection!")
print(f"🎯 Use threshold {threshold:.6f} for inference")
