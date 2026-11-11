import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from .dataset import JSRTDataset
from .transforms import (
    get_train_transforms,
    get_valid_transforms,
    get_test_transforms,
)


# --------------------------------------------------
# Faster R-CNN / RetinaNet collate function
# --------------------------------------------------
def collate_fn(batch):

    return tuple(zip(*batch))


# --------------------------------------------------
# Create WeightedRandomSampler
# --------------------------------------------------
def create_weighted_sampler(csv_file, seed=42):

    df = pd.read_csv(
        csv_file,
        na_values=["?"]
    )

    # Build label column if missing
    if "label" not in df.columns:

        df["label"] = (
            df["state"] != "non-nodule"
        ).astype(int)

    # One label per image
    image_labels = df["label"].astype(int).tolist()

    counts = Counter(image_labels)

    print("=" * 60)
    print("Training Image Distribution")
    print("=" * 60)
    print(f"Negative : {counts[0]}")
    print(f"Positive : {counts[1]}")
    print("=" * 60)

    # Inverse-frequency class weights
    class_weights = {
        0: 1.0 / counts[0],
        1: 1.0 / counts[1],
    }

    sample_weights = [
        class_weights[label]
        for label in image_labels
    ]

    # Seed the sampler
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    return sampler


# --------------------------------------------------
# Build all dataloaders
# --------------------------------------------------
def create_dataloaders(
    train_csv,
    valid_csv,
    test_csv,
    image_dir,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    seed=42,
):

    # persistent_workers requires workers
    persistent_workers = (
        persistent_workers and num_workers > 0
    )

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    train_dataset = JSRTDataset(
        train_csv,
        image_dir,
        transforms=get_train_transforms(),
    )

    valid_dataset = JSRTDataset(
        valid_csv,
        image_dir,
        transforms=get_valid_transforms(),
    )

    test_dataset = JSRTDataset(
        test_csv,
        image_dir,
        transforms=get_test_transforms(),
    )

    # --------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------
    sampler = create_weighted_sampler(
        train_csv,
        seed=seed,
    )

    # --------------------------------------------------
    # Training loader
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Validation loader
    # --------------------------------------------------
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Test loader
    # --------------------------------------------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )
