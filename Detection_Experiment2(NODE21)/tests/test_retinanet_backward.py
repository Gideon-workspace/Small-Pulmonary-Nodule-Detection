import torch

from datasets.dataloader import create_dataloaders
from models.retinanet import get_retinanet

print("=" * 60)
print("Loading dataloader...")
print("=" * 60)

train_loader, _, _ = create_dataloaders(
    train_csv="data/train.csv",
    valid_csv="data/valid.csv",
    test_csv="data/test.csv",
    image_dir="data/images",
)

print("=" * 60)
print("Loading RetinaNet...")
print("=" * 60)

device = torch.device("cpu")

model = get_retinanet()
model.to(device)
model.train()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
)

images, targets = next(iter(train_loader))

images = [img.to(device) for img in images]

targets = [
    {
        k: v.to(device)
        for k, v in t.items()
    }
    for t in targets
]

print("=" * 60)
print("Forward pass")
print("=" * 60)

loss_dict = model(images, targets)

loss = sum(loss_dict.values())

print("Total loss:", loss.item())

print("=" * 60)
print("Backward pass")
print("=" * 60)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("✓ Backward successful")
print("✓ Optimizer step successful")