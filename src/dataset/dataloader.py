
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset.dataset import DeepGlobeDataset
from src.dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_dataloaders(
    dataset_path,
    batch_size=8,
    num_workers=2,
    test_size=0.2,
    random_state=42,
):

    train_dir = os.path.join(dataset_path, "train")

    image_paths = sorted(
        [
            os.path.join(train_dir, file)
            for file in os.listdir(train_dir)
            if file.endswith("_sat.jpg")
        ]
    )

    mask_paths = [
        path.replace("_sat.jpg", "_mask.png")
        for path in image_paths
    ]

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_dataset = DeepGlobeDataset(
        train_images,
        train_masks,
        transform=get_train_transforms(),
    )

    val_dataset = DeepGlobeDataset(
        val_images,
        val_masks,
        transform=get_val_transforms(),
    )

    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        common_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args,
    )

    return train_loader, val_loader
