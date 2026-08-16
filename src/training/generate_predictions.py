
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset.cached_dataloader import get_cached_dataloaders
from src.models.unet import UNet


# ============================================================
# CONFIG
# ============================================================

CACHE_PATH = "/content/DeepGlobe_cache"

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/GeoVision-AI/"
    "checkpoints/baseline/best_model.pth"
)

OUTPUT_DIR = "outputs/predictions"

NUM_CLASSES = 7
BATCH_SIZE = 8
NUM_WORKERS = 2

CLASS_NAMES = [
    "Urban",
    "Agriculture",
    "Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown",
]

# DeepGlobe RGB visualization colors
COLORS = np.array([
    [0, 255, 255],       # Urban
    [255, 255, 0],       # Agriculture
    [255, 0, 255],       # Rangeland
    [0, 255, 0],         # Forest
    [0, 0, 255],         # Water
    [255, 255, 255],     # Barren
    [0, 0, 0],           # Unknown
], dtype=np.uint8)


# ============================================================
# MASK -> RGB
# ============================================================

def mask_to_rgb(mask):
    mask = np.asarray(mask)
    return COLORS[mask]


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("GeoVision-AI — PREDICTION GENERATION")
    print("=" * 70)

    print("Device:", device)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "N/A"),
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    _, val_loader = get_cached_dataloaders(
        cache_path=CACHE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_size=0.2,
        random_state=42,
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------------
    # COLLECT PREDICTIONS
    # --------------------------------------------------------

    collected = []

    with torch.no_grad():

        for images, masks in val_loader:

            images_gpu = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                images_gpu
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            ).cpu()

            for i in range(
                images.shape[0]
            ):

                collected.append(
                    (
                        images[i].cpu(),
                        masks[i].cpu(),
                        predictions[i],
                    )
                )

                if len(collected) >= 6:
                    break

            if len(collected) >= 6:
                break

    # --------------------------------------------------------
    # GENERATE SIX SAMPLES
    # --------------------------------------------------------

    print("\nGenerating samples...")

    for index, (
        image,
        ground_truth,
        prediction,
    ) in enumerate(
        collected,
        start=1,
    ):

        image = image.numpy()

        # Convert CHW -> HWC
        if image.shape[0] == 3:
            image = np.transpose(
                image,
                (1, 2, 0),
            )

        # Normalize image for visualization
        image = image - image.min()

        if image.max() > 0:
            image = image / image.max()

        ground_truth = ground_truth.numpy()
        prediction = prediction.numpy()

        gt_rgb = mask_to_rgb(
            ground_truth
        )

        pred_rgb = mask_to_rgb(
            prediction
        )

        # ----------------------------------------------------
        # PLOT
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        axes[0].imshow(image)
        axes[0].set_title(
            "Original Image",
            fontsize=13,
        )

        axes[1].imshow(gt_rgb)
        axes[1].set_title(
            "Ground Truth",
            fontsize=13,
        )

        axes[2].imshow(pred_rgb)
        axes[2].set_title(
            "U-Net Prediction",
            fontsize=13,
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"GeoVision-AI — Sample {index}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{index}.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )

    print("\n" + "=" * 70)
    print("Prediction generation complete.")
    print("=" * 70)

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
