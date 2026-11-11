import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .backbone import get_backbone


def get_faster_rcnn(
    num_classes=2,
    checkpoint_path=None,
    use_lip_stem=True,
    lip_layers=(2,),
    post_k=1,
):
    """
    Faster R-CNN using the medical-pretrained
    TorchXRayVision ResNet50 + FPN.

    Parameters
    ----------
    num_classes : int
        Background + pulmonary nodule.
        Default = 2.

    checkpoint_path : str or None
        Path to a previously trained Faster R-CNN checkpoint.
        If None, no checkpoint is loaded.

    use_lip_stem : bool
        If True:
            Replace the ResNet MaxPool with LIPModule.

        If False:
            Keep the original ResNet MaxPool.

    Returns
    -------
    model : torchvision FasterRCNN
    """

    # --------------------------------------------------
    # Backbone
    # --------------------------------------------------
    #
    # use_lip_stem=True
    #     -> ResNet + LIPModule + FPN
    #
    # use_lip_stem=False
    #     -> Original ResNet MaxPool + FPN
    #
    # --------------------------------------------------
    backbone = get_backbone(
        use_lip_stem=use_lip_stem,
	lip_layers=lip_layers,
	post_k=post_k
    )

    # --------------------------------------------------
    # Build Faster R-CNN
    # --------------------------------------------------
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
    )

    # --------------------------------------------------
    # Keep NODE21 images at original resolution
    # 1024 × 1024
    # --------------------------------------------------
    model.transform = GeneralizedRCNNTransform(

        min_size=(1024,),

        max_size=1024,

        image_mean=[0.0],

        image_std=[1.0],

    )

    # --------------------------------------------------
    # Load previously learned checkpoint
    # --------------------------------------------------
    #if checkpoint_path is not None:

        #print("=" * 60)
        #print("Loading previously learned Faster R-CNN checkpoint")
        #print("=" * 60)

        #checkpoint = torch.load(
            #checkpoint_path,
            #map_location="cpu"
        #)

        #model.load_state_dict(
            #checkpoint
        #)

        #print("✓ Checkpoint loaded")

    # --------------------------------------------------
    # Return model
    # --------------------------------------------------
    return model


if __name__ == "__main__":

    print("=" * 60)
    print("Creating Faster R-CNN...")
    print("=" * 60)

    model = get_faster_rcnn(
        use_lip_stem=True
    )

    print(model)

    print("\nModel created successfully!")
