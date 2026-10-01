import torch
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .backbone import get_backbone


def get_retinanet(num_classes=2,checkpoint_path =None,use_lip_stem=True,lip_layers=(2,),post_k=1):
    """
    RetinaNet using the medical pretrained
    TorchXRayVision ResNet50 + FPN.

    Classes
    -------
    0 : background
    1 : pulmonary nodule
    """

    backbone = get_backbone(use_lip_stem=use_lip_stem,lip_layers=lip_layers,post_k=post_k)

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
	
    )

    # --------------------------------------------------
    # Keep original NODE21 resolution (1024 × 1024)
    # --------------------------------------------------
    model.transform = GeneralizedRCNNTransform(
        min_size=1024,
        max_size=1024,
        image_mean=[0.0],
        image_std=[1.0],
    )


    # --------------------------------------------------
    # 4. Load previously learned checkpoint
    # --------------------------------------------------

    #if checkpoint_path is not None:

        #print("=" * 60)
        #print("Loading previously learned checkpoint")
        #print("=" * 60)

        #checkpoint = torch.load(
            #checkpoint_path,
            #map_location="cpu"
        #)

        #model.load_state_dict(checkpoint)

        #print("✓ Checkpoint loaded")


    # --------------------------------------------------
    # 5. Unfreeze EVERYTHING
    # --------------------------------------------------

    #for param in model.parameters():
        #param.requires_grad = True

    #print("✓ Entire RetinaNet pipeline is trainable")

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("Creating RetinaNet...")
    print("=" * 60)

    model = get_retinanet()

    print(model)

    print("\nModel created successfully!")
