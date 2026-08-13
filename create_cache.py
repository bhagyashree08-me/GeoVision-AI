
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = "/content/DeepGlobe/train"
CACHE_DIR = "/content/DeepGlobe_cache"

IMAGE_SIZE = 512

os.makedirs(CACHE_DIR, exist_ok=True)

image_cache = os.path.join(CACHE_DIR, "images")
mask_cache = os.path.join(CACHE_DIR, "masks")

os.makedirs(image_cache, exist_ok=True)
os.makedirs(mask_cache, exist_ok=True)


# ============================================================
# FIND DATA
# ============================================================

image_paths = sorted([
    os.path.join(SOURCE_DIR, f)
    for f in os.listdir(SOURCE_DIR)
    if f.endswith("_sat.jpg")
])

print("Images found:", len(image_paths))


# ============================================================
# CACHE
# ============================================================

for image_path in tqdm(
    image_paths,
    desc="Creating cache"
):

    filename = os.path.basename(image_path)

    image_id = filename.replace(
        "_sat.jpg",
        ""
    )

    mask_path = os.path.join(
        SOURCE_DIR,
        f"{image_id}_mask.png"
    )

    image_out = os.path.join(
        image_cache,
        f"{image_id}.npy"
    )

    mask_out = os.path.join(
        mask_cache,
        f"{image_id}.npy"
    )

    # Skip if already cached
    if (
        os.path.exists(image_out)
        and os.path.exists(mask_out)
    ):
        continue

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask = Image.open(
        mask_path
    ).convert("RGB")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        image_out,
        image
    )

    np.save(
        mask_out,
        mask
    )


print("\nCache creation completed.")

print(
    "Cached images:",
    len(os.listdir(image_cache))
)

print(
    "Cached masks:",
    len(os.listdir(mask_cache))
)

print(
    "Cache location:",
    CACHE_DIR
)
