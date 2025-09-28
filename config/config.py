import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_PATH = os.path.join(BASE_DIR, "src", "saved_models")  # <-- renamed for consistency

LOGS_PATH = os.path.join(BASE_DIR, "logs")  # <-- add this line

# Dataset files
KDD_TRAIN = os.path.join(RAW_PATH, "KDDTrain+.txt")
KDD_TEST = os.path.join(RAW_PATH, "KDDTest+.txt")
