
"""
Training engine (Faster R-CNN).

Coordinates:
    • Training
    • Validation
    • Learning-rate scheduler
    • Saving the best model

Author: Mashego Gideon Mabeloane
"""

import os
import torch

from train_one_epoch import train_one_epoch
from validate import validate


def fit(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path="weights/fasterrcnn_baseline_best.pth",
):
    best_loss = float("inf")

    # Ensure the checkpoint directory exists
    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_box_loss": [],
        "valid_loss": [],
        "valid_cls_loss": [],
        "valid_box_loss": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):

        print("=" * 60)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 60)

        # --------------------------------------
        # Training
        # --------------------------------------
        (
            train_loss,
            train_cls_loss,
            train_box_loss,
        ) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # --------------------------------------
        # Validation
        # --------------------------------------
        (
            valid_loss,
            valid_cls_loss,
            valid_box_loss,
        ) = validate(
            model=model,
            dataloader=valid_loader,
            device=device,
        )

        # Record learning rate used during this epoch
        current_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------
        # Save best model
        # --------------------------------------
        if valid_loss < best_loss:
            best_loss = valid_loss

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(f"Best model saved to {save_path}")
            print(f"Best validation loss: {best_loss:.4f}")

        # --------------------------------------
        # Update learning-rate scheduler
        # --------------------------------------
        # For ReduceLROnPlateau, pass validation loss.
        # For StepLR, call scheduler.step() without a metric.
        if isinstance(
            scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        next_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------
        # Print losses and learning rates
        # --------------------------------------
        print(f"Train Total Loss       : {train_loss:.4f}")
        print(f"  Classification       : {train_cls_loss:.4f}")
        print(f"  Box Regression       : {train_box_loss:.4f}")

        print(f"Valid Total Loss       : {valid_loss:.4f}")
        print(f"  Classification       : {valid_cls_loss:.4f}")
        print(f"  Box Regression       : {valid_box_loss:.4f}")

        print(f"LR used this epoch     : {current_lr:.2e}")
        print(f"LR for next epoch      : {next_lr:.2e}")
        print(f"Best validation loss   : {best_loss:.4f}")

        # --------------------------------------
        # Store history
        # --------------------------------------
        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_box_loss"].append(train_box_loss)

        history["valid_loss"].append(valid_loss)
        history["valid_cls_loss"].append(valid_cls_loss)
        history["valid_box_loss"].append(valid_box_loss)

        history["learning_rate"].append(current_lr)

        print()

    return history
