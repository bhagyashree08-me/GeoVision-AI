import torch

from src.training.metrics import (
    pixel_accuracy,
    mean_iou,
    dice_score,
)


def main():
    num_classes = 7

    # --------------------------------------------------
    # Test 1: Perfect prediction
    # --------------------------------------------------

    targets = torch.tensor([
        [
            [0, 1],
            [2, 3]
        ]
    ])

    logits = torch.full(
        (1, num_classes, 2, 2),
        -10.0
    )

    # Make the correct class have the highest logit
    for row in range(2):
        for col in range(2):
            class_id = targets[0, row, col]
            logits[0, class_id, row, col] = 10.0

    accuracy = pixel_accuracy(logits, targets)
    iou = mean_iou(
        logits,
        targets,
        num_classes=num_classes
    )
    dice = dice_score(
        logits,
        targets,
        num_classes=num_classes
    )

    print("PERFECT PREDICTION")
    print("------------------")
    print("Pixel Accuracy :", accuracy)
    print("Mean IoU       :", iou)
    print("Dice Score     :", dice)

    assert abs(accuracy - 1.0) < 1e-6
    assert abs(iou - 1.0) < 1e-6
    assert abs(dice - 1.0) < 1e-6

    # --------------------------------------------------
    # Test 2: Completely wrong prediction
    # --------------------------------------------------

    wrong_targets = torch.tensor([
        [
            [0, 1],
            [2, 3]
        ]
    ])

    wrong_logits = torch.full(
        (1, num_classes, 2, 2),
        -10.0
    )

    # Predict class 4 for every pixel
    wrong_logits[:, 4, :, :] = 10.0

    wrong_accuracy = pixel_accuracy(
        wrong_logits,
        wrong_targets
    )

    wrong_iou = mean_iou(
        wrong_logits,
        wrong_targets,
        num_classes=num_classes
    )

    wrong_dice = dice_score(
        wrong_logits,
        wrong_targets,
        num_classes=num_classes
    )

    print("\nWRONG PREDICTION")
    print("----------------")
    print("Pixel Accuracy :", wrong_accuracy)
    print("Mean IoU       :", wrong_iou)
    print("Dice Score     :", wrong_dice)

    assert wrong_accuracy == 0.0

    print("\nMetric tests PASSED.")


if __name__ == "__main__":
    main()