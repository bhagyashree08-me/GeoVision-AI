import os

import torch

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
            exist_ok=True
        )

        self.best_val_loss = float("inf")

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for images, masks in self.train_loader:

            images = images.to(self.device)
            masks = masks.to(self.device).long()

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                masks
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = (
            running_loss / len(self.train_loader)
        )

        return epoch_loss

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_accuracy = 0.0

        num_batches = len(self.val_loader)

        for images, masks in self.val_loader:

            images = images.to(self.device)
            masks = masks.to(self.device).long()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                masks
            )

            running_loss += loss.item()

            total_iou += mean_iou(
                outputs,
                masks,
                self.num_classes
            )

            total_dice += dice_score(
                outputs,
                masks,
                self.num_classes
            )

            total_accuracy += pixel_accuracy(
                outputs,
                masks
            )

        return {
            "loss": running_loss / num_batches,
            "miou": total_iou / num_batches,
            "dice": total_dice / num_batches,
            "accuracy": total_accuracy / num_batches,
        }

    def save_checkpoint(
        self,
        epoch,
        val_loss,
    ):

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    def fit(self, epochs):

        history = []

        for epoch in range(1, epochs + 1):

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            train_loss = self.train_one_epoch()

            metrics = self.validate()

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Val Loss   : {metrics['loss']:.4f}"
            )

            print(
                f"mIoU       : {metrics['miou']:.4f}"
            )

            print(
                f"Dice       : {metrics['dice']:.4f}"
            )

            print(
                f"Accuracy   : {metrics['accuracy']:.4f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **metrics,
                }
            )

            if metrics["loss"] < self.best_val_loss:

                self.best_val_loss = metrics["loss"]

                self.save_checkpoint(
                    epoch,
                    metrics["loss"]
                )

        return history