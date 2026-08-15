
import torch
from torch.optim import Adam

from src.dataset.cached_dataloader import get_cached_dataloaders
from src.models.unet import UNet
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_PATH = "/content/DeepGlobe_cache"

BATCH_SIZE = 8

NUM_WORKERS = 2

NUM_CLASSES = 7

LEARNING_RATE = 1e-4

# FINAL TRAINING
EPOCHS = 10

CHECKPOINT_DIR = "outputs/checkpoints_weighted"


def main():

    # ========================================================
    # CUDA OPTIMIZATION
    # ========================================================

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)

    print("GeoVision-AI")

    print("=" * 70)

    print(
        "\nDevice:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # ========================================================
    # DATA
    # ========================================================

    train_loader, val_loader = get_cached_dataloaders(
        cache_path=CACHE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(
        "\nTraining batches   :",
        len(train_loader),
    )

    print(
        "Validation batches :",
        len(val_loader),
    )

    print(
        "Batch size         :",
        BATCH_SIZE,
    )

    print(
        "Workers            :",
        NUM_WORKERS,
    )

    print(
        "Epochs             :",
        EPOCHS,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES,
    ).to(device)

    # ========================================================
    # LOSS
    # ========================================================

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=NUM_CLASSES,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    history = trainer.fit(
        epochs=EPOCHS,
    )

    print("\nTraining completed.")

    return history


if __name__ == "__main__":
    main()
