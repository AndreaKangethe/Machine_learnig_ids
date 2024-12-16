#data_loader.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_string_column(col):
    """Cleans string columns by removing extra spaces and quotes."""
    return col.apply(lambda x: x.strip().replace("'", "").replace('"', '') if isinstance(x, str) else x)

def load_and_clean_arff(file_path):
    """
    Cleans the ARFF file by removing extra quotes or spaces and loads it.
    """
    from scipy.io import arff
    with open(file_path, 'r') as f:
        content = f.readlines()

    # Clean lines that may have extra quotes or spaces
    cleaned_content = []
    for line in content:
        line = re.sub(r" 'icmp'", 'icmp', line)
        line = re.sub(r" 'tcp'", 'tcp', line)
        line = re.sub(r" 'udp'", 'udp', line)
        cleaned_content.append(line)

    # Convert cleaned content into a string and load directly into pandas
    from io import StringIO
    cleaned_data = ''.join(cleaned_content)
    cleaned_file = StringIO(cleaned_data)
    data, meta = arff.loadarff(cleaned_file)
    df = pd.DataFrame(data)

    # Clean all string columns
    df = df.apply(lambda col: clean_string_column(col) if col.dtype == 'O' else col)
    df = df.fillna(0)  # Handle missing values
    
    return df

def load_and_clean_csv(file_path):
    """
    Loads and cleans the CSV file.
    """
    df = pd.read_csv(file_path)
    df = df.apply(lambda col: clean_string_column(col) if col.dtype == 'O' else col)
    df = df.fillna(0)  # Handle missing values
    return df

def load_data(file_path):
    """
    Loads and cleans the data, either ARFF or CSV file.
    """
    if file_path.endswith('.arff'):
        print(f"Loading ARFF file from {file_path}...")
        return load_and_clean_arff(file_path)
    elif file_path.endswith('.csv'):
        print(f"Loading CSV file from {file_path}...")
        return load_and_clean_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

def split_data(df):
    """
    Splits the dataframe into features (X) and labels (y), then into training and testing sets.
    """
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Labels
    return train_test_split(X, y, test_size=0.2, random_state=42)
