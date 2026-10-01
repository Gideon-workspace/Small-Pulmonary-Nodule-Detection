"""
Medical pretrained ResNet50 + Feature Pyramid Network (FPN)

This backbone is shared by:
    • Faster R-CNN
    • RetinaNet

LIP (Local Importance-based Pooling) can replace any of the downsampling steps:
    • stem MaxPool                (use_lip_stem=True)
    • first block of layer2/3/4   (lip_layers=(2,), (2, 3), (2, 3, 4), ...)

Requires lip_module_Lswaps.py (apply_lip) in the same directory/package.

Author: Mashego Gideon Mabeloane
"""

import torch
import torchxrayvision as xrv

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# Import apply_lip (handles both package imports and direct script execution)
try:
    from .lip_module import apply_lip
except ImportError:
    from lip_module import apply_lip


def get_backbone(
    use_lip_stem: bool = True,
    lip_layers=(2,),
    post_k: int = 1,
) -> BackboneWithFPN:
    """ 
    Returns
    -------
    BackboneWithFPN

    Parameters
    ----------
    use_lip_stem : bool, default=False
        If True, replaces the stem's MaxPool2d with LIPModule (learnable pooling).
        If False, keeps the standard MaxPool2d.

    lip_layers : iterable, default=(2,)
        Which layers get their downsampling (first block) replaced with LIP.
        (2,)        -> only layer2
        (2, 3, 4)   -> layer2, layer3 and layer4
        ()          -> none (pure baseline when use_lip_stem is also False)

    post_k : int, default=3
        Kernel size of the stride-1 conv after LIP in the residual branch.
        3 reuses the pretrained conv2 weights (recommended with pretrained
        weights); 1 follows the official LIP repo (weights trained from scratch).

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
    # Swap downsampling steps for LIP (weights are already loaded,
    # so apply_lip can reuse the pretrained conv / shortcut weights)
    # --------------------------------------------------
    lip_layers = tuple(lip_layers)

    apply_lip(
        resnet,
        stem_conv=False,          # 7x7 stride-2 conv1 stays as is
        stem_pool=use_lip_stem,   # stem MaxPool -> LIPModule (64 channels)
        layers=lip_layers,        # first block of each listed layer
        post_k=post_k,
    )

    if use_lip_stem:
        print("✓ Stem MaxPool replaced with LIPModule (64 channels)")
    else:
        print("✓ Keeping standard MaxPool2d")

    if lip_layers:
        names = ", ".join(f"layer{l}" if isinstance(l, int) else str(l)
                          for l in lip_layers)
        print(f"✓ LIP downsampling in: {names} (post_k={post_k})")
    else:
        print("✓ No LIP in residual layers")

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
    # Test execution: only layer2 replaced with LIP
    backbone = get_backbone(use_lip_stem=False, lip_layers=(2,)).eval()

    print("\nRunning backbone test...\n")

    x = torch.randn(1, 1, 1024, 1024)
    with torch.no_grad():
        outputs = backbone(x)

    print("=" * 60)
    print("Feature Map Output Shapes")
    print("=" * 60)

    for idx, (name, feature) in enumerate(outputs.items()):
        print(f"P{idx+2} (level {name}): {list(feature.shape)}")

    print("\n✓ Backbone working correctly with LIP in layer2!")

        # gradient reaches the LIP head in layer2
    b = get_backbone(lip_layers=(2,))
    blk = b.body.layer2[0]
    print(type(blk).__name__)                  # expect LIPDownBottleneck
    out = b(torch.randn(1, 1, 512, 512))
    sum(o.mean() for o in out.values()).backward()
    head = blk.conv2[0].postprocessing.conv.weight
    print("LIP head gets gradient:", bool(head.grad.abs().sum() > 0))   # expect True
