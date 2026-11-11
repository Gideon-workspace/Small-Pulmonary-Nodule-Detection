"""
Main training script.

Author: Mashego Gideon Mabeloane
"""

import random
import numpy as np
import torch
import pandas as pd

from datasets.dataloader import create_dataloaders
#from models.faster_rcnn import get_faster_rcnn
from models.retinanet import get_retinanet
# from models.freeze import setup_freeze_schedule
from optimizer import get_optimizer
from engine import fit


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    print(f"Using device : {device}")
    print(f"Random seed  : {seed}")
    print("=" * 60)

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_csv="data/train.csv",
        valid_csv="data/val.csv",
        test_csv="data/test.csv",
        image_dir="data/images",
        seed=seed,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    #model = get_faster_rcnn(use_lip_stem=True)

    model = get_retinanet(
        use_lip_stem=True
    )

    # --------------------------------------------------
    # Move model to device
    # --------------------------------------------------
    model.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer, scheduler = get_optimizer(model)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=50,
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "Retinanet_training_LIP1_history.csv",
        index=False,
    )

    print("Training history saved.")


if __name__ == "__main__":
    main()
