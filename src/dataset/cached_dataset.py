
import numpy as np
import torch

from torch.utils.data import Dataset

from src.dataset.mask_utils import rgb_to_mask


class CachedDeepGlobeDataset(Dataset):

    def __init__(
        self,
        image_paths,
        mask_paths,
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = np.load(
            self.image_paths[index]
        )

        mask = np.load(
            self.mask_paths[index]
        )

        # RGB mask -> class IDs
        mask = rgb_to_mask(mask)

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        else:

            image = torch.from_numpy(
                image
            ).permute(2, 0, 1).float() / 255.0

            mask = torch.from_numpy(
                mask
            ).long()

        return image, mask
