
import os
import json

import torch
import numpy as np

from src.dataset.cached_dataloader import get_cached_dataloaders
from src.models.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_PATH = "/content/DeepGlobe_cache"

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/GeoVision-AI/"
    "checkpoints/baseline/best_model.pth"
)

OUTPUT_DIR = "outputs/evaluation"

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_CLASSES = 7

CLASS_NAMES = [
    "Urban land",
    "Agriculture land",
    "Rangeland",
    "Forest",
    "Water",
    "Barren land",
    "Unknown",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, targets, num_classes):
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    valid = (
        (targets >= 0)
        & (targets < num_classes)
    )

    targets = targets[valid]
    predictions = predictions[valid]

    indices = (
        targets * num_classes
        + predictions
    )

    confusion += torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(
        num_classes,
        num_classes,
    )

    intersection = torch.diag(confusion).float()

    ground_truth = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()

    union = (
        ground_truth
        + predicted
        - intersection
    )

    iou = intersection / union.clamp(min=1)

    dice = (
        2.0 * intersection
        / (
            ground_truth
            + predicted
        ).clamp(min=1)
    )

    accuracy = (
        intersection.sum()
        / confusion.sum().clamp(min=1)
    )

    return (
        accuracy.item(),
        iou.numpy(),
        dice.numpy(),
    )


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
    print("GeoVision-AI — MODEL EVALUATION")
    print("=" * 70)

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "N/A"),
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get("val_loss", "N/A"),
    )

    # ========================================================
    # DATA
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # INFERENCE
    # ========================================================

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                masks.cpu()
            )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy, iou, dice = calculate_metrics(
        predictions,
        targets,
        NUM_CLASSES,
    )

    miou = float(
        np.mean(iou)
    )

    mean_dice = float(
        np.mean(dice)
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"mIoU     : {miou:.4f}"
    )

    print(
        f"Dice     : {mean_dice:.4f}"
    )

    print("\nPer-class metrics")
    print("-" * 70)

    results = {
        "accuracy": accuracy,
        "miou": miou,
        "dice": mean_dice,
        "classes": {},
    }

    for class_id in range(NUM_CLASSES):

        print(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]:<18} "
            f"IoU = {iou[class_id]:.4f} | "
            f"Dice = {dice[class_id]:.4f}"
        )

        results["classes"][
            str(class_id)
        ] = {
            "name": CLASS_NAMES[class_id],
            "iou": float(iou[class_id]),
            "dice": float(dice[class_id]),
        }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_results.json",
    )

    with open(
        results_path,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)

    print(
        "Saved evaluation results to:"
    )

    print(
        os.path.abspath(results_path)
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
