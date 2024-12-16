#main.py
import torch
from model import MyModel
from data_loader import load_data, split_data
from training import train_model
from utility import save_model, load_model
import pandas as pd
import numpy as np
import os  # Import os module to interact with the filesystem
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Import evaluation metrics

# Main function
def main():
    # File paths for your dataset
    file_path = "/home/dre/All projects all language/PycharmProjects/MlIds/data/raw/archive (2)/Wednesday-workingHours.pcap_ISCX.csv"  # Change to your dataset path
    model_save_path = "saved_models/intrusion_detection_model.pth"
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the data
    try:
        print("Loading and preprocessing data...")
        data = load_data(file_path)
        
        # Check for NaN or Inf values and handle them
        if data.isna().any().any() or (data == np.inf).any().any():
            print("Warning: NaN or Inf values detected in data.")
            data = data.replace([np.inf, -np.inf], np.nan)  # Replace Inf with NaN
            data = data.fillna(0)  # Replace NaN with 0 or choose another strategy

        # Split data into training and testing
        X_train, X_test, y_train, y_test = split_data(data)
        print("Data loaded and split successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Model parameters
    input_size = X_train.shape[1]
    num_classes = len(y_train.unique())

    # Initialize the model
    model = MyModel(input_size=input_size, num_classes=num_classes)

    # Train the model
    try:
        print("Starting training...")
        model, scaler, label_encoder = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            epochs=20,
            batch_size=128,
            learning_rate=0.001,
            patience=5,
            device=device,
            model_save_path=model_save_path
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save the model
    try:
        save_model(model, model_save_path)
    except Exception as e:
        print(f"Error saving the model: {e}")

    # Example: Load the saved model for inference
    try:
        print("Loading saved model for inference...")
        loaded_model = MyModel(input_size=input_size, num_classes=num_classes)
        load_model(loaded_model, model_save_path)
        print("Model loaded and ready for inference.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    # Evaluate the model on the test set
    try:
        print("Evaluating model on the test set...")
        # Set the model to evaluation mode
        loaded_model.eval()
        
        # Make predictions
        with torch.no_grad():  # No gradients needed for evaluation
            y_pred = loaded_model(torch.tensor(X_test.values).float().to(device))
            y_pred = y_pred.argmax(dim=1)  # Get the class with the highest probability

        # Calculate metrics
        y_test = y_test.to(device)
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        precision = precision_score(y_test.cpu(), y_pred.cpu(), average='weighted')
        recall = recall_score(y_test.cpu(), y_pred.cpu(), average='weighted')
        f1 = f1_score(y_test.cpu(), y_pred.cpu(), average='weighted')

        # Print evaluation results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
