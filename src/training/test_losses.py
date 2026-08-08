import torch

from src.training.losses import DiceLoss, CombinedLoss


NUM_CLASSES = 7
BATCH_SIZE = 2
HEIGHT = 512
WIDTH = 512


def main():
    # Fake model output
    logits = torch.randn(
        BATCH_SIZE,
        NUM_CLASSES,
        HEIGHT,
        WIDTH
    )

    # Fake segmentation masks
    targets = torch.randint(
        0,
        NUM_CLASSES,
        (BATCH_SIZE, HEIGHT, WIDTH)
    )

    dice_loss = DiceLoss(
        num_classes=NUM_CLASSES
    )

    combined_loss = CombinedLoss(
        num_classes=NUM_CLASSES
    )

    dice_value = dice_loss(
        logits,
        targets
    )

    combined_value = combined_loss(
        logits,
        targets
    )

    print("Logits shape        :", logits.shape)
    print("Targets shape       :", targets.shape)
    print("Dice Loss           :", dice_value.item())
    print("Combined Loss       :", combined_value.item())

    assert torch.isfinite(dice_value)
    assert torch.isfinite(combined_value)

    print("\nLoss test PASSED.")


if __name__ == "__main__":
    main()