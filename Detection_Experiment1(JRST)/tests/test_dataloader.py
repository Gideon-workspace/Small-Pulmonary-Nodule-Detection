import torch

from datasets.dataloader import create_dataloaders


def main():

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_csv="data/train.csv",
        valid_csv="data/val.csv",
        test_csv="data/test.csv",
        image_dir="data/images",
        num_workers=0,          # use 0 for the first test, then try 4
    )

    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    images, targets = next(iter(train_loader))

    print(f"Batch size : {len(images)}")
    print()

    for i, (image, target) in enumerate(zip(images, targets)):

        print(f"Image {i}")

        print("Shape :", image.shape)
        print("Dtype :", image.dtype)

        print("Boxes :", target["boxes"].shape)
        print("Labels:", target["labels"].shape)

        print("-" * 40)


if __name__ == "__main__":

    main()