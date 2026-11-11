import torch
import torchxrayvision as xrv

from torchvision.models.detection.backbone_utils import BackboneWithFPN

# --------------------------------------------------
# Load medical backbone
# --------------------------------------------------

xrv_model = xrv.models.ResNet(
    weights="resnet50-res512-all"
)

backbone = xrv_model.model

# --------------------------------------------------
# Wrap with FPN
# --------------------------------------------------

fpn = BackboneWithFPN(

    backbone=backbone,

    return_layers={
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    },

    in_channels_list=[
        256,
        512,
        1024,
        2048,
    ],

    out_channels=256,

)

print("=" * 60)
print("Backbone + FPN created successfully!")
print("=" * 60)

# --------------------------------------------------
# Dummy input
# --------------------------------------------------

x = torch.randn(
    1,
    1,
    1024,
    1024,
)

outputs = fpn(x)

print("\nFeature Maps\n")

for k, v in outputs.items():

    print(k, v.shape)