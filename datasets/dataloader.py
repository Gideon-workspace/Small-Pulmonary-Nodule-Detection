from collections import Counter

import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from .dataset import Node21Dataset
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
def create_weighted_sampler(csv_file):

    df = pd.read_csv(csv_file)

    groups = df.groupby("img_name")

    image_labels = []

    for _, records in groups:

        image_labels.append(int(records.iloc[0]["label"]))

    counts = Counter(image_labels)

    print("=" * 60)
    print("Training Image Distribution")
    print("=" * 60)
    print(f"Negative : {counts[0]}")
    print(f"Positive : {counts[1]}")
    print("=" * 60)

    class_weights = {

        0: 1.0 / counts[0],
        1: 1.0 / counts[1],

    }

    sample_weights = [

        class_weights[label]

        for label in image_labels

    ]

    sampler = WeightedRandomSampler(

        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,

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
    batch_size=8,
    num_workers=8, # num_workers=4,
    pin_memory = True,
    persistent_workers = True,

):

    train_dataset = Node21Dataset(

        train_csv,
        image_dir,
        transforms=get_train_transforms(),

    )

    valid_dataset = Node21Dataset(

        valid_csv,
        image_dir,
        transforms=get_valid_transforms(),

    )

    test_dataset = Node21Dataset(

        test_csv,
        image_dir,
        transforms=get_test_transforms(),

    )

    sampler = create_weighted_sampler(train_csv)

    train_loader = DataLoader(

        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,

    )

    valid_loader = DataLoader(

        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers,
        collate_fn=collate_fn,

    )

    test_loader = DataLoader(

        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers,
        collate_fn=collate_fn,

    )

    return (

        train_loader,
        valid_loader,
        test_loader,

    )
