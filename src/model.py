#model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyModel, self).__init__()
        
        # First fully connected layer (input layer)
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization after first layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer with 50% probability
        
        # Second fully connected layer (hidden layer)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization after second layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer with 50% probability
        
        # Third fully connected layer (output layer)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Forward pass with ReLU activation
        x = torch.relu(self.bn1(self.fc1(x)))  # Apply batch normalization and ReLU
        x = self.dropout1(x)  # Apply dropout
        
        x = torch.relu(self.bn2(self.fc2(x)))  # Apply batch normalization and ReLU
        x = self.dropout2(x)  # Apply dropout
        
        x = self.fc3(x)  # Final output layer
        return x
