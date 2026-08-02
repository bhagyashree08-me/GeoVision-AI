import numpy as np

# RGB -> Class ID
COLOR_MAP = {
    (0, 255, 255): 0,      # Urban land
    (255, 255, 0): 1,      # Agriculture land
    (255, 0, 255): 2,      # Rangeland
    (0, 255, 0): 3,        # Forest
    (0, 0, 255): 4,        # Water
    (255, 255, 255): 5,    # Barren land
    (0, 0, 0): 6           # Unknown
}


def rgb_to_mask(mask):
    """
    Convert RGB mask (H,W,3) into class-index mask (H,W)
    """
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for rgb, class_id in COLOR_MAP.items():
        matches = np.all(mask == rgb, axis=-1)
        label_mask[matches] = class_id

    return label_mask