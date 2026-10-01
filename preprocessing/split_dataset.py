from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("NODE21 DATASET PREPROCESSING")
print("=" * 60)

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print(f"\nProject directory : {PROJECT_ROOT}")
print(f"Data directory    : {DATA_DIR}")

# --------------------------------------------------
# Load metadata
# --------------------------------------------------
print("\n[1/5] Loading metadata...")

metadata_path = DATA_DIR / "metadata.csv"
df = pd.read_csv(metadata_path)

print(f"✓ Loaded {len(df)} annotations.")

# --------------------------------------------------
# Convert image names
# --------------------------------------------------
print("\n[2/5] Converting image names (.mha → .png)...")

df["img_name"] = df["img_name"].str.replace(".mha", ".png", regex=False)

# --------------------------------------------------
# Create bounding boxes
# --------------------------------------------------
print("\n[3/5] Creating Faster R-CNN bounding boxes...")

df["xmin"] = df["x"]
df["ymin"] = df["y"]
df["xmax"] = df["x"] + df["width"]
df["ymax"] = df["y"] + df["height"]

df = df[
    [
        "img_name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
    ]
]

print("✓ Bounding boxes created.")

# --------------------------------------------------
# Split by unique images
# --------------------------------------------------
print("\n[4/5] Splitting dataset...")

image_df = (
    df.groupby("img_name")["label"]
      .max()
      .reset_index()
)

train_imgs, temp_imgs = train_test_split(
    image_df,
    test_size=0.30,
    stratify=image_df["label"],
    random_state=42,
)

val_imgs, test_imgs = train_test_split(
    temp_imgs,
    test_size=0.50,
    stratify=temp_imgs["label"],
    random_state=42,
)

train_df = df[df["img_name"].isin(train_imgs["img_name"])]
val_df = df[df["img_name"].isin(val_imgs["img_name"])]
test_df = df[df["img_name"].isin(test_imgs["img_name"])]

print("✓ Dataset successfully split.")

# --------------------------------------------------
# Save CSV files
# --------------------------------------------------
print("\n[5/5] Saving CSV files...")

train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("✓ train.csv saved.")
print("✓ val.csv saved.")
print("✓ test.csv saved.")

# --------------------------------------------------
# Statistics
# --------------------------------------------------
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Training annotations   : {len(train_df)}")
print(f"Validation annotations : {len(val_df)}")
print(f"Testing annotations    : {len(test_df)}")

print()

print("Training Images")
print(train_imgs["label"].value_counts().rename({
    0: "Negative",
    1: "Positive"
}))

print()

print("Validation Images")
print(val_imgs["label"].value_counts().rename({
    0: "Negative",
    1: "Positive"
}))

print()

print("Testing Images")
print(test_imgs["label"].value_counts().rename({
    0: "Negative",
    1: "Positive"
}))

print("\n" + "=" * 60)
print("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)