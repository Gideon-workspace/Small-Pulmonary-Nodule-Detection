from pathlib import Path

import cv2
import pandas as pd
import torch
import torchxrayvision as xrv

from torch.utils.data import Dataset


class Node21Dataset(Dataset):
    """
    NODE21 Dataset.

    Compatible with:

        • Faster R-CNN
        • RetinaNet

    Returns
    -------
    image : Tensor
        Shape = [1, H, W]

    target : dict
    """

    def __init__(

        self,
        csv_file,
        image_dir,
        transforms=None,

    ):

        self.df = pd.read_csv(csv_file)

        self.image_dir = Path(image_dir)

        self.transforms = transforms

        # Group annotations belonging to one image
        self.groups = self.df.groupby("img_name")

        # Unique images
        self.image_names = list(self.groups.groups.keys())

        print("=" * 60)
        print("NODE21 DATASET LOADED")
        print("=" * 60)
        print(f"CSV File : {csv_file}")
        print(f"Images   : {len(self.image_names)}")
        print(f"Rows     : {len(self.df)}")
        print("=" * 60)

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):

        # --------------------------------------------------
        # Image
        # --------------------------------------------------
        image_name = self.image_names[idx]

        image_path = self.image_dir / image_name

        image = cv2.imread(

            str(image_path),

            cv2.IMREAD_GRAYSCALE,

        )

        if image is None:

            raise FileNotFoundError(

                f"Image not found: {image_path}"

            )

        # --------------------------------------------------
        # Annotations
        # --------------------------------------------------
        records = self.groups.get_group(image_name)

        # --------------------------------------------------
        # Negative Image
        # --------------------------------------------------
        if records.iloc[0]["label"] == 0:

            boxes = torch.zeros(

                (0, 4),

                dtype=torch.float32,

            )

            labels = torch.zeros(

                (0,),

                dtype=torch.int64,

            )

        # --------------------------------------------------
        # Positive Image
        # --------------------------------------------------
        else:

            boxes = records[

                [

                    "xmin",

                    "ymin",

                    "xmax",

                    "ymax",

                ]

            ].values

            boxes = torch.as_tensor(

                boxes,

                dtype=torch.float32,

            )

            labels = torch.ones(

                (len(boxes),),

                dtype=torch.int64,

            )

        # --------------------------------------------------
        # Albumentations
        # --------------------------------------------------
        if self.transforms is not None:

            transformed = self.transforms(

                image=image,

                bboxes=boxes.numpy(),

                labels=labels.numpy(),

            )

            image = transformed["image"]

            boxes = torch.as_tensor(

                transformed["bboxes"],

                dtype=torch.float32,

            )

            labels = torch.as_tensor(

                transformed["labels"],

                dtype=torch.int64,

            )

        # --------------------------------------------------
        # TorchXRayVision Normalization
        # --------------------------------------------------
        image = xrv.utils.normalize(

            image,

            maxval=255,

        )

        # --------------------------------------------------
        # Convert to Tensor
        # --------------------------------------------------
        image = torch.from_numpy(

            image

        ).float()

        # Add channel dimension

        image = image.unsqueeze(0)

        # --------------------------------------------------
        # Target
        # --------------------------------------------------
        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor([idx]),

        }

        return image, target