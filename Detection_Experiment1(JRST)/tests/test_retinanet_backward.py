import torch

from datasets.dataloader import create_dataloaders
from models.retinanet import get_retinanet


print("Loading data...")

train_loader, _, _ = create_dataloaders(
    train_csv="data/train.csv",
    valid_csv="data/val.csv",
    test_csv="data/test.csv",
    image_dir="data/images",
    batch_size=1,
    num_workers=0,
)

print("Loading model...")

model = get_retinanet(use_lip_stem=False)
model.train()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
)

print("Getting one image...")

images, targets = next(iter(train_loader))

# Use only the first image
images = [images[0]]
targets = [targets[0]]

print("Image:", images[0].shape)

print("Forward...")

loss_dict = model(images, targets)

# If we reach here, the forward pass has completed
print("✓ FORWARD PASS COMPLETED")

loss = sum(loss_dict.values())

print("Loss:", loss.item())

print("============================================================")
print("Starting BACKWARD pass...")
print("============================================================")

optimizer.zero_grad()

loss.backward()

print("✓ BACKWARD PASS COMPLETED")

print("Starting optimizer step...")

optimizer.step()

print("✓ OPTIMIZER STEP COMPLETED")