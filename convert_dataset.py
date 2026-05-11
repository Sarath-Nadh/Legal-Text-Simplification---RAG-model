from datasets import load_dataset
import pandas as pd

# Load dataset
ds = load_dataset("viber1/indian-law-dataset")

# Convert to pandas
df = ds['train'].to_pandas()

# 🔥 IMPORTANT: print columns
print(df.columns)

# Save only first 300 rows
df.head(300).to_csv("data/legal_dataset.csv", index=False)

print("Dataset saved successfully!")