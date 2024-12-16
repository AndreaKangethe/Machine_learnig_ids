#utility
import os
import torch

def save_model(model, file_path):
    """
    Save the trained model to the specified file path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f'Model successfully saved to {file_path}')

def load_model(model, file_path):
    """
    Load a model's state_dict from the specified file path.
    """
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        model.eval()
        print(f'Model successfully loaded from {file_path}')
    else:
        print(f'Error: Model file not found at {file_path}')
