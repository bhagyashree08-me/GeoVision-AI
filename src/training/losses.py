import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.
    """

    def __init__(self, num_classes=7, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W]
            targets: [B, H, W]

        Returns:
            Dice loss
        """

        probabilities = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes
        )

        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        ).float()

        dims = (0, 2, 3)

        intersection = torch.sum(
            probabilities * targets_one_hot,
            dims
        )

        denominator = torch.sum(
            probabilities + targets_one_hot,
            dims
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss.
    """

    def __init__(
        self,
        num_classes=7,
        ce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.cross_entropy = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(
            num_classes=num_classes
        )

    def forward(self, logits, targets):
        ce = self.cross_entropy(logits, targets)
        dice = self.dice_loss(logits, targets)

        loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
        )

        return loss