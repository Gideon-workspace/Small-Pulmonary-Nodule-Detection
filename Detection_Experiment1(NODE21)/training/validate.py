"""
Validate the detector.

Works with:
    • Faster R-CNN

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
    Validate one epoch.

    Returns
    -------
    tuple
        (Average total loss,
         Average classification loss,
         Average box regression loss)
    """

    # --------------------------------------------------
    # Keep training mode active so Faster R-CNN
    # returns its loss dictionary (torchvision detection
    # models only compute losses in train() mode).
    #
    # BatchNorm and Dropout are kept in evaluation mode.
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

    running_loss = 0.0

    running_cls_loss = 0.0

    running_box_loss = 0.0

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
                targets
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
            # Group losses (same grouping as training)
            # ------------------------------------------
            cls_loss = (
                loss_classifier
                + loss_objectness
            )

            box_loss = (
                loss_box_reg
                + loss_rpn_box_reg
            )

            # ------------------------------------------
            # Total loss
            # ------------------------------------------
            loss = (
                loss_classifier
                + loss_box_reg
                + loss_objectness
                + loss_rpn_box_reg
            )

            # ------------------------------------------
            # Accumulate
            # ------------------------------------------
            running_loss += loss.item()

            running_cls_loss += cls_loss.item()

            running_box_loss += box_loss.item()

            # ------------------------------------------
            # Progress bar
            # ------------------------------------------
            progress_bar.set_postfix(

                loss=f"{loss.item():.4f}",

                cls=f"{cls_loss.item():.4f}",

                box=f"{box_loss.item():.4f}",

                avg_loss=f"{running_loss / (progress_bar.n + 1):.4f}",

            )

    # --------------------------------------------------
    # Average losses
    # --------------------------------------------------
    return (

        running_loss / len(dataloader),

        running_cls_loss / len(dataloader),

        running_box_loss / len(dataloader),

    )
