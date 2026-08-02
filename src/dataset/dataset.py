import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from src.dataset.mask_utils import rgb_to_mask


class DeepGlobeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(
            Image.open(self.image_paths[idx]).convert("RGB")
        )

        mask = np.array(
            Image.open(self.mask_paths[idx]).convert("RGB")
        )

        mask = rgb_to_mask(mask)

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask