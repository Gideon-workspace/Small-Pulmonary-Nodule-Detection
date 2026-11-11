from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv

from torch.utils.data import Dataset


class JSRTDataset(Dataset):
    """
    JSRT Dataset (nodule / non-nodule).

    Compatible with:

        • Faster R-CNN
        • RetinaNet

    CSV columns used
    ----------------
    study_id : image file name (e.g. JPCLN001.png)
    x, y     : nodule centre in pixels (original 2048 x 2048 image)
    size     : nodule diameter in mm
    label    : 1 = nodule (malignant or benign), 0 = non-nodule

    Returns
    -------
    image : Tensor
        Shape = [1, H, W]

    target : dict
        boxes    [N, 4] (xmin, ymin, xmax, ymax) in pixels of the loaded image
        labels   [N]    (1 = nodule)
        image_id [1]
    """

    # JSRT original images: 2048 x 2048 px, 0.175 mm / pixel
    ORIG_SIZE = 2048
    MM_PER_PIXEL = 0.175

    def __init__(
        self,
        csv_file,
        image_dir,
        transforms=None,
    ):

        self.df = pd.read_csv(csv_file, na_values=["?"])

        # Safety: build the label column if it is missing
        if "label" not in self.df.columns:

            self.df["label"] = (self.df["state"] != "non-nodule").astype(int)

        self.image_dir = Path(image_dir)

        self.transforms = transforms

        # One row per image in JSRT
        self.image_names = self.df["study_id"].tolist()

        print("=" * 60)
        print("JSRT DATASET LOADED")
        print("=" * 60)
        print(f"CSV File : {csv_file}")
        print(f"Images   : {len(self.image_names)}")
        print(f"Nodules  : {int((self.df['label'] == 1).sum())}")
        print(f"Negatives: {int((self.df['label'] == 0).sum())}")
        print("=" * 60)

    def __len__(self):

        return len(self.image_names)

    def __getitem__(self, idx):

        # --------------------------------------------------
        # Image
        # --------------------------------------------------
        record = self.df.iloc[idx]

        image_name = record["study_id"]

        image_path = self.image_dir / image_name

        image = cv2.imread(

            str(image_path),

            cv2.IMREAD_GRAYSCALE,

        )

        if image is None:

            raise FileNotFoundError(

                f"Image not found: {image_path}"

            )

        h, w = image.shape[:2]

        # --------------------------------------------------
        # Negative Image
        # --------------------------------------------------
        if record["label"] == 0:

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
        # (x, y, size) -> (xmin, ymin, xmax, ymax)
        # --------------------------------------------------
        else:

            # Scale in case the PNGs were resized from 2048 x 2048
            scale_x = w / self.ORIG_SIZE
            scale_y = h / self.ORIG_SIZE

            cx = float(record["x"]) * scale_x
            cy = float(record["y"]) * scale_y

            # diameter mm -> pixels (original resolution) -> current resolution
            diameter_px = float(record["size"]) / self.MM_PER_PIXEL

            half_w = diameter_px * scale_x / 2.0
            half_h = diameter_px * scale_y / 2.0

            xmin = max(cx - half_w, 0.0)
            ymin = max(cy - half_h, 0.0)
            xmax = min(cx + half_w, w - 1.0)
            ymax = min(cy + half_h, h - 1.0)

            boxes = torch.as_tensor(

                [[xmin, ymin, xmax, ymax]],

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

                np.array(transformed["bboxes"]),

                dtype=torch.float32,

            ).reshape(-1, 4)

            labels = torch.as_tensor(

                np.array(transformed["labels"]),

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