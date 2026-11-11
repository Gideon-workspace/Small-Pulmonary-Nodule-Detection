"""
Test one forward pass through RetinaNet.

Goal:
-----
Verify that

Dataset
    ↓
DataLoader
    ↓
Medical Backbone
    ↓
FPN
    ↓
RetinaNet

works correctly.

No training occurs.
"""

import torch

from datasets.dataloader import create_dataloaders
from models.retinanet import get_retinanet
from models.faster_rcnn import get_faster_rcnn


def main():

    print("=" * 60)
    print("Loading dataloader...")
    print("=" * 60)

    train_loader, _, _ = create_dataloaders(
        train_csv="data/train.csv",
        valid_csv="data/val.csv",
        test_csv="data/test.csv",
        image_dir="data/images",
        batch_size=1,
        num_workers=0,
    )

    print("=" * 60)
    print("Loading Faster R-CNN...")
    print("=" * 60)

    model = get_faster_rcnn(use_lip_stem = True)

    model.train()

    print("=" * 60)
    print("Getting one positive image...")
    print("=" * 60)

    for images, targets in train_loader:

        if len(targets[0]["boxes"]) > 0:
            break

    print(f"Number of images : {len(images)}")
    print(f"Image shape      : {images[0].shape}")
    print(f"Image dtype      : {images[0].dtype}")

    print()
    print("Ground-truth boxes:")
    print(targets[0]["boxes"])

    print()
    print("Ground-truth labels:")
    print(targets[0]["labels"])

    print()

    print("=" * 60)
    print("Forward pass...")
    print("=" * 60)

    with torch.no_grad():

        losses = model(images, targets)

    print()

    print("=" * 60)
    print("Loss Dictionary")
    print("=" * 60)

    total_loss = 0.0

    for name, value in losses.items():

        print(f"{name:25s}: {value.item():.6f}")

        total_loss += value

    print("-" * 60)
    print(f"Total Loss              : {total_loss.item():.6f}")

    print()
    print("✓ Forward pass successful!")


if __name__ == "__main__":
    main()