from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("JSRT DATASET PREPROCESSING")
print("=" * 60)

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Change this if your metadata file has a different name
METADATA_FILE = "jsrt_metadata.csv"

print(f"\nProject directory : {PROJECT_ROOT}")
print(f"Data directory    : {DATA_DIR}")

# --------------------------------------------------
# Load metadata (robust reading)
# --------------------------------------------------
print("\n[1/5] Loading metadata...")

metadata_path = DATA_DIR / METADATA_FILE

if not metadata_path.exists():

    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

# sep=None + engine="python" auto-detects comma / semicolon / tab
# encoding="utf-8-sig" removes the hidden BOM Excel adds to CSV files
df = pd.read_csv(
    metadata_path,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
    na_values=["?"],
)

# Clean header names: remove spaces and make lowercase
df.columns = df.columns.str.strip().str.lower()

print(f"✓ Loaded {len(df)} images.")
print(f"  Columns found: {df.columns.tolist()}")

# --------------------------------------------------
# Validate columns
# --------------------------------------------------
required = ["study_id", "x", "y", "size"]

missing_cols = [c for c in required if c not in df.columns]

if missing_cols:

    raise ValueError(
        f"Missing columns: {missing_cols}\n"
        f"Columns found: {df.columns.tolist()}\n"
        "If you see ONE long column name, the file was not split into "
        "columns. In Excel use Data > Text to Columns (Comma), then "
        "save as CSV (Comma delimited)."
    )

# --------------------------------------------------
# Check / create the label column
# (0 = non-nodule, 1 = nodule: malignant or benign)
# --------------------------------------------------
print("\n[2/5] Checking labels...")

if "label" in df.columns:

    print("✓ 'label' column found.")

elif "state" in df.columns:

    df["label"] = (df["state"].str.strip() != "non-nodule").astype(int)

    print("✓ 'label' column created from 'state'.")

else:

    raise ValueError(
        "Neither 'label' nor 'state' column found. "
        f"Columns found: {df.columns.tolist()}"
    )

# Keep a 'state' column for later reporting (malignant / benign)
if "state" not in df.columns:

    df["state"] = ""

df["label"] = df["label"].astype(int)

# Sanity check: every nodule image needs x, y and size
nodules = df[df["label"] == 1]

n_missing = nodules[["x", "y", "size"]].isna().any(axis=1).sum()

if n_missing > 0:

    raise ValueError(
        f"{n_missing} nodule rows are missing x, y or size."
    )

print("✓ Nodule rows contain x, y and size.")

# Boxes are NOT precomputed here: JSRTDataset builds them from
# (x, y, size) and rescales them to the loaded image resolution.

# --------------------------------------------------
# Keep the columns needed for training
# --------------------------------------------------
print("\n[3/5] Selecting columns...")

df = df[
    [
        "study_id",
        "x",
        "y",
        "size",
        "state",
        "label",
    ]
]

print("✓ Columns selected.")

# --------------------------------------------------
# Split by unique images
# --------------------------------------------------
print("\n[4/5] Splitting dataset...")

image_df = (
    df.groupby("study_id")["label"]
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

train_df = df[df["study_id"].isin(train_imgs["study_id"])]
val_df = df[df["study_id"].isin(val_imgs["study_id"])]
test_df = df[df["study_id"].isin(test_imgs["study_id"])]

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

print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Testing images    : {len(test_df)}")

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