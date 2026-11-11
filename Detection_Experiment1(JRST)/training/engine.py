"""
Training engine (RetinaNet).

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
    save_path="weights/retinanet_LIP1_best.pth",

):


    best_loss = float("inf")

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
        # Train
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
        # Validate
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

        # --------------------------------------
        # Record the learning rate used THIS epoch,
        # then update the scheduler
        # --------------------------------------
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        # --------------------------------------
        # Print losses
        # --------------------------------------
        print(f"Train Total Loss       : {train_loss:.4f}")
        print(f"  Classification       : {train_cls_loss:.4f}")
        print(f"  Box Regression       : {train_box_loss:.4f}")

        print(f"Valid Total Loss       : {valid_loss:.4f}")
        print(f"  Classification       : {valid_cls_loss:.4f}")
        print(f"  Box Regression       : {valid_box_loss:.4f}")

        print(f"Learning rate          : {current_lr:.6f}")

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

        # --------------------------------------
        # Save best model
        # --------------------------------------
        if valid_loss < best_loss:

            best_loss = valid_loss

            torch.save(

                model.state_dict(),

                save_path,

            )

            print(f"✓ Best model saved to {save_path}")

        print()

    return history
