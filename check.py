import pandas as pd

df = pd.read_csv("data/processed/kdd_combined_processed.csv")

# Column 42 is labels
label_col = df.columns[42]

# Binary conversion: 0 (normal) vs 1 (attack)
df["binary_label"] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

print("Binary label counts:")
print(df["binary_label"].value_counts())

# Save CSV for balanced split
df.to_csv("data/processed/kdd_combined_processed_binary.csv", index=False)
