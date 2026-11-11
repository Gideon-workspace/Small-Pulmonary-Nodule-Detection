"""
Medical pretrained ResNet50 + Feature Pyramid Network (FPN)

This backbone is shared by:
    • Faster R-CNN
    • RetinaNet

Author: Mashego Gideon Mabeloane
"""

import torch
import torchxrayvision as xrv

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# Import LIP module (assumes LIP_module.py is in the same directory/package)
# Import LIP module (handles both direct script execution and package imports)
try:
    from .lip_module import LIPModule
except ImportError:
    from lip_module import LIPModule

def get_backbone(use_lip_stem: bool = True) -> BackboneWithFPN:
    """
    Returns
    -------
    BackboneWithFPN

    Parameters
    ----------
    use_lip_stem : bool, default=True
        If True, replaces the stem's MaxPool2d with LIPModule (learnable pooling).
        If False, preserves standard MaxPool2d for baseline A/B testing.

    Output feature maps:
        0 -> layer1 (256 ch, 1/4 resolution)
        1 -> layer2 (512 ch, 1/8 resolution)
        2 -> layer3 (1024 ch, 1/16 resolution)
        3 -> layer4 (2048 ch, 1/32 resolution)

    out_channels = 256 (FPN projection)
    """

    print("=" * 60)
    print("Loading Medical Pretrained ResNet50")
    print("=" * 60)

    # --------------------------------------------------
    # Load TorchXRayVision model
    # --------------------------------------------------
    xrv_model = xrv.models.ResNet(
        weights="resnet50-res512-all"
    )

    # Underlying torchvision ResNet50
    resnet = xrv_model.model

    print("✓ Medical weights loaded")

    # --------------------------------------------------
    # Optionally replace stem MaxPool with LIP
    # --------------------------------------------------
    if use_lip_stem:
        # Stem output after conv1 + bn1 + relu has 64 channels
        resnet.maxpool = LIPModule(channels=64, kernel_size=3, stride=2, padding=1)
        print("✓ Stem MaxPool replaced with LIPModule (64 channels)")
    else:
        print("✓ Keeping standard MaxPool2d (Baseline mode)")

    # --------------------------------------------------
    # Return feature maps instead of classifier
    # --------------------------------------------------
    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    body = IntermediateLayerGetter(
        resnet,
        return_layers=return_layers,
    )

    # --------------------------------------------------
    # Number of channels produced by each ResNet stage
    # --------------------------------------------------
    in_channels_list = [
        256,   # layer1
        512,   # layer2
        1024,  # layer3
        2048,  # layer4
    ]

    out_channels = 256

    # --------------------------------------------------
    # Build FPN
    # --------------------------------------------------
    backbone = BackboneWithFPN(
        backbone=body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    print("✓ Feature Pyramid Network attached")
    print("✓ Backbone ready for detection")
    print("=" * 60)

    return backbone


if __name__ == "__main__":
    # Test execution
    backbone = get_backbone(use_lip_stem=True)

    print("\nRunning backbone test...\n")

    x = torch.randn(1, 1, 1024, 1024)
    outputs = backbone(x)

    print("=" * 60)
    print("Feature Map Output Shapes")
    print("=" * 60)

    for idx, (name, feature) in enumerate(outputs.items()):
        print(f"P{idx+2} (level {name}): {list(feature.shape)}")

    print("\n✓ Backbone working correctly with LIP stem!")
