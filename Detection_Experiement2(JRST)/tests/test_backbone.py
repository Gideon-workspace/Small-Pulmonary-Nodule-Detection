import torchxrayvision as xrv

model = xrv.models.ResNet(weights="resnet50-res512-all")

print("=" * 60)
print("Children of model.model")
print("=" * 60)

for name, module in model.model.named_children():
    print(name)