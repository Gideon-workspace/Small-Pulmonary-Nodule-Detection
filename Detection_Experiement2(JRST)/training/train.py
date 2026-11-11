
"""
Main training script.

Author: Mashego Gideon Mabeloane
"""

print("PYTHON SCRIPT STARTED", flush=True)

import random

import numpy as np
import pandas as pd
import torch

print("TORCH IMPORTED", flush=True)

from datasets.dataloader import create_dataloaders
print("CREATE_DATALOADER IMPORTED", flush=True)

from models.faster_rcnn import get_faster_rcnn
#from models.retinanet import get_retinanet
print("RETINANET IMPORTED", flush=True)

from optimizer import get_optimizer
from engine import fit

print("ALL IMPORTS DONE, ENTERING THE MAIN", flush=True)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # --------------------------------------------------
    # Seed
    # --------------------------------------------------
    seed = 42
    set_seed(seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------
    print("STEP 1: Creating dataloaders...", flush=True)

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_csv="data/train.csv",
        valid_csv="data/val.csv",
        test_csv="data/test.csv",
        image_dir="data/images",
    )

    print("STEP 1 DONE: Dataloaders created.", flush=True)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    print("STEP 2: Creating FasterRCNN...", flush=True)

    model = get_faster_rcnn(
        use_lip_stem=False,
        lip_layers=(),
        post_k=1,
    )

    print("STEP 2 DONE: FasterRCNN created.", flush=True)

    model.to(device)

    print("STEP 3 DONE: Model moved to device.", flush=True)

    # --------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------
    print("STEP 4: Creating optimizer...", flush=True)

    optimizer, scheduler = get_optimizer(model)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    print("STEP 5: Starting training...", flush=True)

    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=50,
        save_path="weights/fasterrcnn_baseline_best.pth",
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------
    history_df = pd.DataFrame(history)

    history_path = "Fasterrcnn_baseline_training_history.csv"

    history_df.to_csv(
        history_path,
        index=False,
    )

    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
