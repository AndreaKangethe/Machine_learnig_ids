import pandas as pd
import numpy as np

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data for the ML-IDS model.
    - Fill missing values
    - Encode categorical features
    - Convert all columns to float
    """
    # Fill missing values
    df = df.fillna(0)

    # Encode categorical columns if any
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    # Convert all columns to float
    df = df.astype(float)
    return df


def format_prediction(pred: np.ndarray, label_map: dict = None) -> str:
    """
    Convert model prediction to human-readable label.
    - pred: numpy array or list with predicted label
    - label_map: optional mapping of numeric labels to text labels
    """
    if isinstance(pred, (list, np.ndarray)):
        pred = pred[0]

    if label_map:
        return label_map.get(int(pred), str(pred))

    return str(pred)
