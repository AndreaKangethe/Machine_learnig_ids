import pandas as pd

df = pd.read_csv("mlids/data/processed/kdd_combined_processed.csv")
print(df["label"].value_counts())
