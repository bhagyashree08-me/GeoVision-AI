import torch
from torch.optim import Adam

from src.dataset.dataloader import get_dataloaders
from src.models.unet import UNet
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer


DATASET_PATH = "/content/DeepGlobe"

BATCH_SIZE = 2
NUM_CLASSES = 7
LEARNING_RATE = 1e-4

# Start with 1 epoch for the first real training test.
EPOCHS = 1

CHECKPOINT_DIR = "outputs/checkpoints"


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_loader, val_loader = get_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        "Training batches   :",
        len(train_loader)
    )

    print(
        "Validation batches :",
        len(val_loader)
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES,
    ).to(device)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history = trainer.fit(
        epochs=EPOCHS
    )

    print("\nTraining completed.")

    print("\nHistory:")
    for record in history:
        print(record)


if __name__ == "__main__":
    main()