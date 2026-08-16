
import os
import time

import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.dataset.cached_dataloader import get_cached_dataloaders
from src.models.segformer import SegFormer
from src.training.losses import CombinedLoss


CACHE_PATH = "/content/DeepGlobe_cache"

BATCH_SIZE = 2
NUM_WORKERS = 2

NUM_CLASSES = 7

LEARNING_RATE = 1e-4

EPOCHS = 5

ACCUMULATION_STEPS = 2

CHECKPOINT_DIR = "outputs/checkpoints_segformer"

DRIVE_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/GeoVision-AI/checkpoints/segformer"
)


def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in tqdm(
            loader,
            desc="Validation",
            dynamic_ncols=True,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            ).long()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        DRIVE_CHECKPOINT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("GeoVision-AI — SegFormer-B0")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = (
        get_cached_dataloaders(
            cache_path=CACHE_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    print(
        "Training batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Effective batch size:",
        BATCH_SIZE * ACCUMULATION_STEPS,
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    model = SegFormer(
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = CombinedLoss(
        num_classes=NUM_CLASSES,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    print("SEGFORMER TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            dynamic_ncols=True,
        )

        for step, (images, masks) in enumerate(
            progress,
            start=1,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            ).long()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                )

                loss = (
                    loss
                    / ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            if (
                step % ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
                * ACCUMULATION_STEPS
            )

            progress.set_postfix(
                loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = (
            time.time()
            - epoch_start
        ) / 60

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Time       : {epoch_time:.2f} min"
        )

        print("-" * 70)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scaler_state_dict":
                    scaler.state_dict(),
                "val_loss":
                    val_loss,
            }

            local_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth",
            )

            drive_path = os.path.join(
                DRIVE_CHECKPOINT_DIR,
                "best_model.pth",
            )

            torch.save(
                checkpoint,
                local_path,
            )

            torch.save(
                checkpoint,
                drive_path,
            )

            print(
                "Checkpoint saved:",
                local_path,
            )

            print(
                "Drive backup saved:",
                drive_path,
            )

    print("\n" + "=" * 70)
    print("SEGFORMER 5-EPOCH SANITY RUN COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss,
    )


if __name__ == "__main__":
    main()
