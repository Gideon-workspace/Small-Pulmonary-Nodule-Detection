"""
Train the detector for one epoch.

Works with:

    • Faster R-CNN

Author: Mashego Gideon Mabeloane
"""

import torch
from tqdm.auto import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    """
    Train one epoch.

    Returns
    -------
    tuple
        Average total loss,
        average classification loss,
        average box regression loss.
    """

    max_norm = 10.0

    model.train()

    running_loss = 0.0

    running_cls_loss = 0.0

    running_box_loss = 0.0

    # ------------------------------------------
    # Progress bar
    # ------------------------------------------
    progress_bar = tqdm(

        dataloader,

        desc="Training",

        leave=True,

    )

    # ------------------------------------------
    # Training loop
    # ------------------------------------------
    for images, targets in progress_bar:

        # ------------------------------------------
        # Move images to device
        # ------------------------------------------
        images = [
            image.to(device)
            for image in images
        ]

        # ------------------------------------------
        # Move targets to device
        # ------------------------------------------
        targets = [
            {
                k: v.to(device)
                for k, v in target.items()
            }
            for target in targets
        ]

        # ------------------------------------------
        # Forward pass
        # ------------------------------------------
        loss_dict = model(
            images,
            targets,
        )

        # ------------------------------------------
        # Faster R-CNN losses
        # ------------------------------------------
        loss_classifier = loss_dict[
            "loss_classifier"
        ]

        loss_box_reg = loss_dict[
            "loss_box_reg"
        ]

        loss_objectness = loss_dict[
            "loss_objectness"
        ]

        loss_rpn_box_reg = loss_dict[
            "loss_rpn_box_reg"
        ]

        # ------------------------------------------
        # Group losses
        # ------------------------------------------
        #
        # Classification:
        #   ROI classifier
        #   +
        #   RPN objectness
        #
        cls_loss = (
            loss_classifier
            + loss_objectness
        )

        # ------------------------------------------
        # Box regression:
        #   ROI box regression
        #   +
        #   RPN box regression
        # ------------------------------------------
        box_loss = (
            loss_box_reg
            + loss_rpn_box_reg
        )

        # ------------------------------------------
        # Total Faster R-CNN loss
        # ------------------------------------------
        loss = (
            loss_classifier
            + loss_box_reg
            + loss_objectness
            + loss_rpn_box_reg
        )

        # ------------------------------------------
        # Backprop
        # ------------------------------------------
        optimizer.zero_grad()

        loss.backward()

        # ------------------------------------------
        # Gradient clipping
        # ------------------------------------------
        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=max_norm,
        )

        # ------------------------------------------
        # Optimizer step
        # ------------------------------------------
        optimizer.step()

        # ------------------------------------------
        # Accumulate losses
        # ------------------------------------------
        running_loss += loss.item()

        running_cls_loss += cls_loss.item()

        running_box_loss += box_loss.item()

        # ------------------------------------------
        # Update progress bar
        # ------------------------------------------
        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            cls=f"{cls_loss.item():.4f}",

            box=f"{box_loss.item():.4f}",

            avg_loss=f"{running_loss / (progress_bar.n + 1):.4f}",

        )

    # ------------------------------------------
    # Return average losses
    # ------------------------------------------
    return (

        running_loss / len(dataloader),

        running_cls_loss / len(dataloader),

        running_box_loss / len(dataloader),

    )
