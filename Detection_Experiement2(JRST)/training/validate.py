
"""
Validate the detector.

Works with:
    • torchvision Faster R-CNN
      (loss dict keys:
       "loss_classifier",
       "loss_box_reg",
       "loss_objectness",
       "loss_rpn_box_reg")

Author: Mashego Gideon Mabeloane
"""

import torch
from tqdm.auto import tqdm


def validate(
    model,
    dataloader,
    device,
):
    """
    Validate Faster R-CNN for one epoch.

    Returns
    -------
    tuple
        Average total loss,
        average classification loss,
        average combined box regression loss.
    """

    # --------------------------------------------------
    # Keep training mode active so Faster R-CNN
    # returns its loss dictionary rather than detections.
    #
    # BatchNorm and Dropout remain in evaluation mode.
    # Gradients are disabled below.
    # --------------------------------------------------
    model.train()

    for module in model.modules():

        if isinstance(
            module,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.Dropout,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            ),
        ):
            module.eval()

    # --------------------------------------------------
    # Running losses
    # --------------------------------------------------
    running_loss = 0.0
    running_cls_loss = 0.0
    running_box_loss = 0.0

    # --------------------------------------------------
    # Progress bar
    # --------------------------------------------------
    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=True,
    )

    # --------------------------------------------------
    # No gradients required
    # --------------------------------------------------
    with torch.no_grad():

        for images, targets in progress_bar:

            # Move images to device
            images = [
                image.to(device)
                for image in images
            ]

            # Move targets to device
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
            cls_loss = loss_dict["loss_classifier"]

            box_loss = (
                loss_dict["loss_box_reg"]
                + loss_dict["loss_rpn_box_reg"]
            )

            objectness_loss = loss_dict["loss_objectness"]

            # ------------------------------------------
            # Total Faster R-CNN loss
            # ------------------------------------------
            loss = sum(loss_dict.values())

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
                obj=f"{objectness_loss.item():.4f}",
                avg_loss=f"{running_loss / (progress_bar.n + 1):.4f}",
            )

    # --------------------------------------------------
    # Return average losses
    # --------------------------------------------------
    num_batches = len(dataloader)

    if num_batches == 0:
        raise ValueError("Validation dataloader is empty.")

    return (
        running_loss / num_batches,
        running_cls_loss / num_batches,
        running_box_loss / num_batches,
    )
