"""
NODE21 RetinaNet Baseline vs LIP Testing
=========================================

Loads the best trained RetinaNet baseline and RetinaNet + LIP weights
and evaluates both models on the test set, then runs a paired bootstrap.

Author: Mashego Gideon Mabeloane
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import JSRTDataset
from datasets.transforms import get_test_transforms
from datasets.dataloader import collate_fn

from models.retinanet import get_retinanet
from models.faster_rcnn import get_faster_rcnn
from evaluate import (
    evaluate_model,
    save_results,
    evaluate_models_with_bootstrap,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 2

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
# Saved models
# ------------------------------------------------------------

BASELINE_WEIGHTS_PATH = (
    "weights/fasterrcnn_baseline_best.pth"
)

PROPOSED_WEIGHTS_PATH = (
    "weights/fasterrcnn_LIP1_best.pth"
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

BASELINE_RESULTS_PATH = (
    "testing/fasterrcnn_baseline_valid_evaluation_results.csv"
)

BASELINE_FROC_CURVE_PATH = (
    "testing/fasterrcnn_baseline_valid_froc_curve.csv"
)

RESULTS_PATH = (
    "testing/fasterrcnn_lip_valid_evaluation_results.csv"
)

FROC_CURVE_PATH = (
    "testing/fasterrcnn_lip_valid_froc_curve.csv"
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("NODE21 RETINANET TESTING")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Seed   : {SEED}"
    )

    print("=" * 60)


    # ========================================================
    # 1. Create test dataset
    # ========================================================

    print()
    print("1. Creating test dataset...")
    print("=" * 60)

    test_dataset = JSRTDataset(

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
    # 3. Create model architectures
    # ========================================================

    print()
    print("3. Creating RetinaNet models...")
    print("=" * 60)

    baseline_model = get_faster_rcnn(
        num_classes=2,
        use_lip_stem=False,
    )

    proposed_model = get_faster_rcnn(
        num_classes=2,
        use_lip_stem=True,
    )


    # ========================================================
    # 4. Load best weights
    # ========================================================

    print()
    print("4. Loading trained weights...")
    print("=" * 60)

    baseline_weights_path = Path(
        BASELINE_WEIGHTS_PATH
    )

    if not baseline_weights_path.exists():

        raise FileNotFoundError(
            f"Could not find baseline weights: "
            f"{baseline_weights_path}"
        )

    baseline_model.load_state_dict(
        torch.load(
            baseline_weights_path,
            map_location=DEVICE,
        )
    )

    print(
        f"✓ Baseline RetinaNet loaded: "
        f"{BASELINE_WEIGHTS_PATH}"
    )

    proposed_weights_path = Path(
        PROPOSED_WEIGHTS_PATH
    )

    if not proposed_weights_path.exists():

        raise FileNotFoundError(
            f"Could not find proposed weights: "
            f"{proposed_weights_path}"
        )

    proposed_model.load_state_dict(
        torch.load(
            proposed_weights_path,
            map_location=DEVICE,
        )
    )

    print(
        f"✓ Proposed LIP loaded: "
        f"{PROPOSED_WEIGHTS_PATH}"
    )


    # ========================================================
    # 5. Move models to device
    # ========================================================

    baseline_model.to(DEVICE)
    baseline_model.eval()

    proposed_model.to(DEVICE)
    proposed_model.eval()

    print(
        f"✓ Models moved to {DEVICE}"
    )


    # ========================================================
    # 6. Evaluate
    # ========================================================

    print()
    print("5. Evaluating BASELINE model...")
    print("=" * 60)


    baseline_results = evaluate_model(

        model=baseline_model,

        test_loader=test_loader,

        test_dataset=test_dataset,

        train_csv=TRAIN_CSV,

        device=DEVICE,

        visualization_dir="testing/visualizations/Fasterrcnn_baseline",

        froc_curve_output_file=BASELINE_FROC_CURVE_PATH,

    )


    print()
    print("6. Evaluating PROPOSED LIP model...")
    print("=" * 60)

    proposed_results = evaluate_model(

        model=proposed_model,

        test_loader=test_loader,

        test_dataset=test_dataset,

        train_csv=TRAIN_CSV,

        device=DEVICE,

        visualization_dir="testing/visualizations/Fasterrcnn_proposed",

        froc_curve_output_file=FROC_CURVE_PATH,

    )


    # ========================================================
    # 7. Save results
    # ========================================================

    print()
    print("7. Saving evaluation results...")
    print("=" * 60)

    save_results(

        baseline_results,

        output_file=BASELINE_RESULTS_PATH,

    )

    save_results(

        proposed_results,

        output_file=RESULTS_PATH,

    )


    # ========================================================
    # 8. Paired bootstrap comparison
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING PAIRED BOOTSTRAP")
    print("=" * 70)

    bootstrap_results = (
        evaluate_models_with_bootstrap(

            baseline_model=baseline_model,

            proposed_model=proposed_model,

            test_loader=test_loader,

            test_dataset=test_dataset,

            train_csv=TRAIN_CSV,

            device=DEVICE,

            n_bootstrap=2000,

            seed=SEED,

        )
    )


    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    print(
        f"Baseline results saved to : "
        f"{BASELINE_RESULTS_PATH}"
    )

    print(
        f"Proposed results saved to : "
        f"{RESULTS_PATH}"
    )

    print(
        f"Baseline FROC saved to    : "
        f"{BASELINE_FROC_CURVE_PATH}"
    )

    print(
        f"Proposed FROC saved to    : "
        f"{FROC_CURVE_PATH}"
    )

    print(
        "Bootstrap results saved to:"
    )

    print(
        "testing/bootstrap_results.csv"
    )

    print(
        "testing/bootstrap_results_summary.csv"
    )

    print("=" * 60)



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
