
import os
import time

import torch
from tqdm import tqdm

from src.training.metrics import (
    pixel_accuracy,
    mean_iou,
    dice_score,
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes=7,
        checkpoint_dir="outputs/checkpoints",
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device
        self.num_classes = num_classes

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        self.best_val_loss = float("inf")

        # Mixed Precision
        self.use_amp = device.type == "cuda"

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

    # ======================================================
    # TRAIN ONE EPOCH
    # ======================================================

    def train_one_epoch(
        self,
        epoch,
        total_epochs,
    ):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{total_epochs} [Train]",
            dynamic_ncols=True,
            leave=True,
        )

        for images, masks in progress:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            masks = masks.to(
                self.device,
                non_blocking=True,
            ).long()

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # Mixed precision forward pass
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.use_amp,
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    masks,
                )

            # Backpropagation
            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return (
            running_loss
            / len(self.train_loader)
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_accuracy = 0.0

        progress = tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc="Validation",
            dynamic_ncols=True,
            leave=True,
        )

        for images, masks in progress:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            masks = masks.to(
                self.device,
                non_blocking=True,
            ).long()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.use_amp,
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    masks,
                )

            running_loss += loss.item()

            total_iou += mean_iou(
                outputs,
                masks,
                self.num_classes,
            )

            total_dice += dice_score(
                outputs,
                masks,
                self.num_classes,
            )

            total_accuracy += pixel_accuracy(
                outputs,
                masks,
            )

        num_batches = len(self.val_loader)

        return {
            "loss": running_loss / num_batches,
            "miou": total_iou / num_batches,
            "dice": total_dice / num_batches,
            "accuracy": total_accuracy / num_batches,
        }

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        epoch,
        val_loss,
    ):

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "best_model.pth",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        print(
            f"\nCheckpoint saved: {checkpoint_path}"
        )

        # Persistent backup in Google Drive
        drive_checkpoint_dir = "/content/drive/MyDrive/GeoVision-AI/checkpoints"
        os.makedirs(drive_checkpoint_dir, exist_ok=True)

        drive_checkpoint_path = os.path.join(
            drive_checkpoint_dir,
            "best_model.pth",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_loss": val_loss,
            },
            drive_checkpoint_path,
        )

        print(
            f"Drive backup saved: {drive_checkpoint_path}"
        )

    # ======================================================
    # FULL TRAINING
    # ======================================================

    def fit(self, epochs):

        history = []

        total_start = time.time()

        print("\n" + "=" * 95)
        print("                         GeoVision-AI TRAINING")
        print("=" * 95)

        for epoch in range(1, epochs + 1):

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            epoch_start = time.time()

            # Training
            train_loss = self.train_one_epoch(
                epoch,
                epochs,
            )

            # Validation
            metrics = self.validate()

            epoch_time = time.time() - epoch_start

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "miou": metrics["miou"],
                "dice": metrics["dice"],
                "time_min": epoch_time / 60,
            }

            history.append(record)

            print("\n" + "-" * 95)

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {metrics['loss']:.4f}")
            print(f"Accuracy   : {metrics['accuracy']:.4f}")
            print(f"mIoU       : {metrics['miou']:.4f}")
            print(f"Dice       : {metrics['dice']:.4f}")
            print(
                f"Epoch Time : {epoch_time / 60:.2f} min"
            )

            print("-" * 95)

            # Save best model
            if metrics["loss"] < self.best_val_loss:

                self.best_val_loss = metrics["loss"]

                self.save_checkpoint(
                    epoch,
                    metrics["loss"],
                )

        # ==================================================
        # FINAL TABLE
        # ==================================================

        total_time = time.time() - total_start

        print("\n" + "=" * 105)
        print("                         FINAL RESULTS")
        print("=" * 105)

        print(
            f"\nTotal training time: "
            f"{total_time / 60:.2f} minutes\n"
        )

        print(
            f"{'Epoch':<8}"
            f"{'Train Loss':<15}"
            f"{'Val Loss':<15}"
            f"{'Accuracy':<15}"
            f"{'mIoU':<15}"
            f"{'Dice':<15}"
            f"{'Time(min)':<12}"
        )

        print("-" * 105)

        for row in history:

            print(
                f"{row['epoch']:<8}"
                f"{row['train_loss']:<15.4f}"
                f"{row['val_loss']:<15.4f}"
                f"{row['accuracy']:<15.4f}"
                f"{row['miou']:<15.4f}"
                f"{row['dice']:<15.4f}"
                f"{row['time_min']:<12.2f}"
            )

        print("=" * 105)

        # Best results
        best_miou = max(
            history,
            key=lambda x: x["miou"],
        )

        best_dice = max(
            history,
            key=lambda x: x["dice"],
        )

        best_val_loss = min(
            history,
            key=lambda x: x["val_loss"],
        )

        print(
            f"\nBest mIoU      : "
            f"Epoch {best_miou['epoch']} "
            f"→ {best_miou['miou']:.4f}"
        )

        print(
            f"Best Dice      : "
            f"Epoch {best_dice['epoch']} "
            f"→ {best_dice['dice']:.4f}"
        )

        print(
            f"Best Val Loss  : "
            f"Epoch {best_val_loss['epoch']} "
            f"→ {best_val_loss['val_loss']:.4f}"
        )

        print("=" * 105)

        return history
