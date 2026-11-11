"""
Albumentations transforms for JSRT.

This file is responsible ONLY for data augmentation.

Normalization is handled inside dataset.py using the
official TorchXRayVision preprocessing.

Notes
-----
* Boxes are in pascal_voc format (xmin, ymin, xmax, ymax), in pixels.
* JSRT images are 2048 x 2048. Pass image_size (e.g. 1024) to downscale;
  boxes are rescaled automatically by Albumentations.
  Leave image_size=None to keep the original resolution
  (torchvision detection models resize internally anyway).
* Do NOT add ToTensorV2 here: dataset.py normalizes a NumPy array
  and converts to a tensor itself.
"""

import albumentations as A


def _bbox_params():

    return A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
    )


# --------------------------------------------------
# Training transforms
# --------------------------------------------------
def get_train_transforms(image_size=None):

    transforms = []

    if image_size is not None:

        transforms.append(A.Resize(image_size, image_size))

    transforms += [

        # Random horizontal flip
        A.HorizontalFlip(p=0.5),

        # Slight brightness/contrast variation
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.30,
        ),

    ]

    return A.Compose(

        transforms,

        bbox_params=_bbox_params(),

    )


# --------------------------------------------------
# Validation transforms
# --------------------------------------------------
def get_valid_transforms(image_size=None):

    transforms = []

    if image_size is not None:

        transforms.append(A.Resize(image_size, image_size))

    return A.Compose(

        transforms,

        bbox_params=_bbox_params(),

    )


# --------------------------------------------------
# Test transforms
# --------------------------------------------------
def get_test_transforms(image_size=None):

    return get_valid_transforms(image_size)