
"""
NODE21 RetinaNet Testing
========================

Loads the best trained RetinaNet baseline weights and evaluates
the model on the NODE21 test set.

Author: Mashego Gideon Mabeloane
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.dataset import Node21Dataset
from datasets.transforms import get_test_transforms
from datasets.dataloader import collate_fn

from models.retinanet import get_retinanet

from evaluate import (
    evaluate_model,
    save_results,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------

TEST_CSV = "data/test.csv"

TRAIN_CSV = "data/train.csv"

IMAGE_DIR = "data/images"

# ------------------------------------------------------------
# Saved model
# ------------------------------------------------------------

WEIGHTS_PATH = (
    "weights/retinanet_baseline_best.pth"
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

RESULTS_PATH = (
    "testing/evaluation_results.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("NODE21 RETINANET TESTING")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print("=" * 60)


    # ========================================================
    # 1. Create test dataset
    # ========================================================

    print()
    print("1. Creating test dataset...")
    print("=" * 60)

    test_dataset = Node21Dataset(

        csv_file=TEST_CSV,

        image_dir=IMAGE_DIR,

        transforms=get_test_transforms(),

    )

    print(
        f"Test images : "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # 2. Create test DataLoader
    # ========================================================

    print()
    print("2. Creating test DataLoader...")
    print("=" * 60)

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn,

        pin_memory=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),

    )

    print(
        f"Batch size  : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Workers     : "
        f"{NUM_WORKERS}"
    )


    # ========================================================
    # 3. Create model architecture
    # ========================================================

    print()
    print("3. Creating RetinaNet...")
    print("=" * 60)

    model = get_retinanet(
        num_classes=2
    )


    # ========================================================
    # 4. Load best weights
    # ========================================================

    print()
    print("4. Loading trained weights...")
    print("=" * 60)

    weights_path = Path(
        WEIGHTS_PATH
    )

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Could not find weights: "
            f"{weights_path}"
        )

    checkpoint = torch.load(
        weights_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "✓ Best RetinaNet weights loaded"
    )

    print(
        f"✓ Weights: "
        f"{WEIGHTS_PATH}"
    )


    # ========================================================
    # 5. Move model to device
    # ========================================================

    model.to(DEVICE)

    model.eval()

    print(
        f"✓ Model moved to {DEVICE}"
    )


    # ========================================================
    # 6. Evaluate
    # ========================================================

    print()
    print("5. Evaluating model...")
    print("=" * 60)

    results = evaluate_model(

        model=model,

        test_loader=test_loader,

        test_dataset=test_dataset,

        train_csv=TRAIN_CSV,

        device=DEVICE,

    )


    # ========================================================
    # 7. Save results
    # ========================================================

    print()
    print("6. Saving evaluation results...")
    print("=" * 60)

    save_results(

        results,

        output_file=RESULTS_PATH,

    )


    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()

