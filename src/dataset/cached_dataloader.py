
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset.cached_dataset import CachedDeepGlobeDataset
from src.dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_cached_dataloaders(
    cache_path="/content/DeepGlobe_cache",
    batch_size=8,
    num_workers=2,
    test_size=0.2,
    random_state=42,
):

    image_dir = os.path.join(
        cache_path,
        "images"
    )

    mask_dir = os.path.join(
        cache_path,
        "masks"
    )

    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".npy")
    ])

    mask_files = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.endswith(".npy")
    ])

    if len(image_files) != len(mask_files):
        raise RuntimeError(
            f"Image/mask mismatch: "
            f"{len(image_files)} images, "
            f"{len(mask_files)} masks"
        )

    print("Cached images:", len(image_files))
    print("Cached masks :", len(mask_files))

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files,
        mask_files,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_dataset = CachedDeepGlobeDataset(
        train_images,
        train_masks,
        transform=get_train_transforms(),
    )

    val_dataset = CachedDeepGlobeDataset(
        val_images,
        val_masks,
        transform=get_val_transforms(),
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args,
    )

    return train_loader, val_loader
