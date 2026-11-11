"""
Albumentations transforms for NODE21.

This file is responsible ONLY for data augmentation.

Normalization is handled inside dataset.py using the
official TorchXRayVision preprocessing.
"""

import albumentations as A


# --------------------------------------------------
# Training transforms
# --------------------------------------------------
def get_train_transforms():

    return A.Compose(

        [

            # Random horizontal flip
            A.HorizontalFlip(p=0.5),

            # Slight brightness/contrast variation
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.30,
            ),

        ],

        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),

    )


# --------------------------------------------------
# Validation transforms
# --------------------------------------------------
def get_valid_transforms():

    return A.Compose(

        [

        ],

        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),

    )


# --------------------------------------------------
# Test transforms
# --------------------------------------------------
def get_test_transforms():

    return get_valid_transforms()