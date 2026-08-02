from src.dataset.dataloader import get_dataloaders

DATASET_PATH = r"D:\datasets\DeepGlobe"

train_loader, val_loader = get_dataloaders(
    dataset_path=DATASET_PATH,
    batch_size=4
)

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")

images, masks = next(iter(train_loader))

print("Images shape :", images.shape)
print("Masks shape  :", masks.shape)

print("Mask dtype   :", masks.dtype)
print("Unique classes:", masks.unique())