#training.py
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

def preprocess_data(X_train):
    X_train = X_train.fillna(X_train.mean())  # Handle missing values

    # Encode categorical features and scale numerical ones
    for column in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column].astype(str))
        print(f"Encoded column: {column}")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    return X_train, scaler

def train_model(model, X_train, y_train, optimizer=None, scheduler=None, epochs=10, batch_size=64, 
                learning_rate=0.001, patience=3, device='cpu', model_save_path=None):
    X_train, scaler = preprocess_data(X_train)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler is None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss}')
        scheduler.step()

    return model, scaler, label_encoder
