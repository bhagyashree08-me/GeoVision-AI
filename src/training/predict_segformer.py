import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.segformer import SegFormer
from src.dataset.cached_dataloader import get_cached_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    "/content/drive/MyDrive/GeoVision-AI/"
    "checkpoints/segformer/best_model.pth"
)

OUTPUT_DIR = "outputs/predictions_segformer"

NUM_CLASSES = 7
NUM_SAMPLES = 6
BATCH_SIZE = 2

CLASS_NAMES = [
    "Urban",
    "Agriculture",
    "Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown",
]


def denormalize(image):
    """
    Reverse the existing cache normalization approximately
    for visualization only.
    """

    image = image.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = (
        image - image.min()
    ) / (
        image.max()
        - image.min()
        + 1e-8
    )

    return np.clip(
        image,
        0,
        1,
    )


def colorize_mask(mask):
    """
    Convert class IDs into a simple segmentation color map.
    """

    colors = np.array(
        [
            [128, 128, 128],  # Urban
            [255, 255, 0],    # Agriculture
            [255, 165, 0],    # Rangeland
            [0, 128, 0],      # Forest
            [0, 0, 255],      # Water
            [165, 42, 42],    # Barren
            [0, 0, 0],        # Unknown
        ],
        dtype=np.uint8,
    )

    return colors[
        np.clip(
            mask,
            0,
            NUM_CLASSES - 1,
        )
    ]


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("GeoVision-AI — SEGFORMER PREDICTION")
    print("=" * 70)

    print("Device:", DEVICE)

    _, val_loader = get_cached_dataloaders(
        cache_path="/content/DeepGlobe_cache",
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    model = SegFormer(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation loss:",
        checkpoint["val_loss"],
    )

    sample_count = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):

                outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            for i in range(
                images.shape[0]
            ):

                if sample_count >= NUM_SAMPLES:
                    break

                image = denormalize(
                    images[i]
                )

                ground_truth = (
                    masks[i]
                    .cpu()
                    .numpy()
                )

                prediction = (
                    predictions[i]
                    .cpu()
                    .numpy()
                )

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15, 5),
                )

                axes[0].imshow(image)
                axes[0].set_title(
                    "Input Image"
                )

                axes[1].imshow(
                    colorize_mask(
                        ground_truth
                    )
                )
                axes[1].set_title(
                    "Ground Truth"
                )

                axes[2].imshow(
                    colorize_mask(
                        prediction
                    )
                )
                axes[2].set_title(
                    "SegFormer Prediction"
                )

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()

                sample_count += 1

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"sample_{sample_count}.png",
                )

                plt.savefig(
                    output_path,
                    dpi=150,
                    bbox_inches="tight",
                )

                plt.close()

                print(
                    "Saved:",
                    output_path,
                )

            if sample_count >= NUM_SAMPLES:
                break

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Prediction generation complete.")
    print("=" * 70)

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
