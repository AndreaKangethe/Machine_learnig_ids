#evaluation.py
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(data, label_encoder=None, scaler=None):
    """Preprocess the input data."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns

    if label_encoder is None:
        label_encoder = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            data[col] = label_encoder[col].fit_transform(data[col])
    else:
        for col in categorical_columns:
            data[col] = label_encoder[col].transform(data[col])

    if scaler is None:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    else:
        data[numerical_columns] = scaler.transform(data[numerical_columns])

    return data, label_encoder, scaler

def save_encoders_scalers(label_encoder, scaler, encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    """Save the label encoder and scaler to disk."""
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    print(f"LabelEncoder saved to {encoder_path}")
    print(f"Scaler saved to {scaler_path}")

def load_encoders_scalers(encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    """Load the label encoder and scaler from disk."""
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    return label_encoder, scaler

def evaluate_model(model, X_test, y_test, device='cpu', class_names=None):
    """Evaluate the trained model on the test data."""
    X_test, label_encoder, scaler = preprocess_data(X_test)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    correct = (predicted == y_test_tensor).sum().item()
    accuracy = correct / len(y_test_tensor) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test_tensor.cpu(), predicted.cpu(), target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    if class_names:
        plot_confusion_matrix(cm, class_names)

    return accuracy, cm

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix using seaborn."""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
