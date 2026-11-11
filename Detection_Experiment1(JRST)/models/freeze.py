import torch
import torch.nn as nn

def setup_freeze_schedule(
    model: nn.Module, 
    unfreeze_lip: bool = True, 
    unfreeze_fpn: bool = True
) -> None:
    """
    Configures parameter gradients for fine-tuning.

    Freezes:
        • Pretrained ResNet stem (conv1, bn1) and residual blocks (layer1..4)
    Unfreezes:
        • LIPModule inside the stem (if unfreeze_lip=True)
        • FPN projection and lateral layers (if unfreeze_fpn=True)
        • Detection heads (RetinaNet head or Faster R-CNN RPN + RoI heads)
    """
    # Step 1: Freeze all backbone parameters by default
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze LIP Module inside the ResNet stem
    if unfreeze_lip:
        if hasattr(model.backbone.body, "maxpool"):
            for param in model.backbone.body.maxpool.parameters():
                param.requires_grad = True
            print("✓ LIPModule parameters set to TRAINABLE")

    # Step 3: Unfreeze Feature Pyramid Network (FPN)
    if unfreeze_fpn:
        if hasattr(model.backbone, "fpn"):
            for param in model.backbone.fpn.parameters():
                param.requires_grad = True
            print("✓ FPN parameters set to TRAINABLE")

    # Step 4: Ensure Detection Heads remain fully trainable
    head_params = 0
    if hasattr(model, "head"):  # RetinaNet
        for param in model.head.parameters():
            param.requires_grad = True
            head_params += param.numel()
        print("✓ RetinaNet Detection Head set to TRAINABLE")
    elif hasattr(model, "roi_heads"):  # Faster R-CNN
        for param in model.rpn.parameters():
            param.requires_grad = True
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        print("✓ Faster R-CNN RPN & RoI Heads set to TRAINABLE")

    print_trainable_parameters(model)


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints a detailed breakdown of trainable vs. frozen parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("PARAMETER TRAINABILITY SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen Parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":

    from pathlib import Path
    from .retinanet import get_retinanet



    # Instantiate model
    model = get_retinanet(num_classes=2)

    # Apply freeze schedule
    setup_freeze_schedule(model, unfreeze_lip=True, unfreeze_fpn=True)
