import pandas as pd

df = pd.read_csv("data/train.csv")

# Count unique images per label
counts = df.groupby("img_name")["label"].first().value_counts().sort_index()

negative = counts.get(0, 0)
positive = counts.get(1, 0)

print("=" * 40)
print(f"Positive images : {positive}")
print(f"Negative images : {negative}")
print(f"Total images    : {positive + negative}")
print(f"Imbalance ratio : {negative / positive:.2f}:1")
print("=" * 40)