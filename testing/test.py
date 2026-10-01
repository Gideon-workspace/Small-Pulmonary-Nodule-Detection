"""
NODE21 Faster RCNN Validation Testing
===========================

Loads the best trained Faster R-CNN baseline weights and evaluates
the model on the NODE21 test set.

Author: Mashego Gideon Mabeloane
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.dataset import Node21Dataset
from datasets.transforms import get_test_transforms
from datasets.dataloader import collate_fn

#from models.retinanet import get_retinanet
from models.faster_rcnn import get_faster_rcnn
from evaluate import (
    evaluate_model,
    save_results,
    evaluate_models_with_bootstrap,
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

TEST_CSV = "data/valid.csv"

TRAIN_CSV = "data/train.csv"

IMAGE_DIR = "data/images"

# ------------------------------------------------------------
# Saved model
# ------------------------------------------------------------
BASELINE_WEIGHTS_PATH = (
     "weights/fasterrcnn_baseline_best.pth"
)
PROPOSED_WEIGHTS_PATH = (
    "weights/fasterrcnn_LIP2_best.pth"
)



# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

RESULTS_PATH = (
    "testing/fasterrcnn_lip2_valid_evaluation_results.csv"
)

FROC_CURVE_PATH = (
    "testing/fasterrcnn_lip2_valid_froc_curve.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("NODE21 FASTER R-CNN TESTING")
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

    baseline_model = get_faster_rcnn(
        num_classes=2,
	use_lip_stem = False,
	lip_layers=()
    )
 
    proposed_model = get_faster_rcnn(
	num_classes=2,
	use_lip_stem = True,
	lip_layers=(2,),
	post_k=1,
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
            f"Could not find weights: "
            f"{weights_path}"
        )

    checkpoint = torch.load(
        baseline_weights_path,
        map_location=DEVICE,
    )

    baseline_model.load_state_dict(
        checkpoint
    )

    print(
        "✓ Best Baseline RetinaNet weights loaded"
    )

    print(
        f"✓ Weights: "
        f"{BASELINE_WEIGHTS_PATH}"
    )

    # ========================================================
    # 4. Create PROPOSED LIP model
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING PROPOSED LIP RETINANET")
    print("=" * 70)


    proposed_weights_path = Path(
        PROPOSED_WEIGHTS_PATH
    )

    if not proposed_weights_path.exists():
        raise FileNotFoundError(
            f"Proposed weights not found: "
            f"{proposed_weights_path}"
        )

    proposed_checkpoint = torch.load(
        proposed_weights_path,
        map_location=DEVICE,
    )

    proposed_model.load_state_dict(
        proposed_checkpoint
    )

  

    print(
        f"✓ Proposed LIP loaded: "
        f"{PROPOSED_WEIGHTS_PATH}"
    )


    # ========================================================
    # 5. Move model to device
    # ========================================================

    baseline_model.to(DEVICE)
    baseline_model.eval()

    proposed_model.to(DEVICE)
    proposed_model.eval()

    print(
        f"✓ Model moved to {DEVICE}"
    )


    # ========================================================
    # 6. Evaluate
    # =======================================================

    print()
    print("5. Evaluating Baseline model...")
    print("=" * 60)

    baseline_results = evaluate_model(

        model=baseline_model,

        test_loader=test_loader,

        test_dataset=test_dataset,

        train_csv=TRAIN_CSV,

        device=DEVICE,

	visualization_dir="testing/visualizations/Fbaseline",

        froc_curve_output_file="testing/fasterrcnn_baseline_valid_froc_curve.csv",

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
	
	visualization_dir="testing/visualizations/Fproposed",

        froc_curve_output_file=FROC_CURVE_PATH,

    )


    # ========================================================
    # 7. Save results
    # ========================================================

    print()
    print("6. Saving evaluation results...")
    print("=" * 60)

    save_results(

        baseline_results,

        output_file="testing/fasterrcnn_baseline_valid_evaluation_results.csv",
    )

    save_results(

        proposed_results,

        output_file=RESULTS_PATH,

    )


    # ========================================================
    #  Paired bootstrap comparison
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
        f"testing/fasterrcnn_baseline_valid_evaluation_results.csv"
    )

    print(
        f"Proposed results saved to : "
        f"{RESULTS_PATH}"
    )

    print(
        f"Baseline FROC saved to    : "
        f"testing/fasterrcnn_baseline_valid_froc_curve.csv"
    )

    print(
        f"Proposed FROC saved to    : "
        f"{FROC_CURVE_PATH}"
    )

    print(
        "Bootstrap results saved to:"
    )

    print(
        "testing/fasterrcnn_bootstrap_results.csv"
    )

    print(
        "testing/fasterrcnn_bootstrap_results_summary.csv"
    )

    print("=" * 60)



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
