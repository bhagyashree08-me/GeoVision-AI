import torch


def pixel_accuracy(logits, targets):
    """
    Calculate pixel accuracy.

    Args:
        logits:  [B, C, H, W]
        targets: [B, H, W]

    Returns:
        Pixel accuracy as a float.
    """

    predictions = torch.argmax(logits, dim=1)

    correct = (predictions == targets).sum()
    total = targets.numel()

    return (correct.float() / total).item()


def mean_iou(logits, targets, num_classes=7):
    """
    Calculate mean Intersection over Union (mIoU).

    Args:
        logits:      [B, C, H, W]
        targets:     [B, H, W]
        num_classes: Number of segmentation classes.

    Returns:
        Mean IoU across classes that are present in the target.
    """

    predictions = torch.argmax(logits, dim=1)

    ious = []

    for class_id in range(num_classes):
        prediction_class = predictions == class_id
        target_class = targets == class_id

        intersection = (
            prediction_class & target_class
        ).sum().float()

        union = (
            prediction_class | target_class
        ).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

    if not ious:
        return 0.0

    return torch.stack(ious).mean().item()


def dice_score(logits, targets, num_classes=7):
    """
    Calculate mean Dice coefficient across classes
    that are present in the target.

    Args:
        logits:      [B, C, H, W]
        targets:     [B, H, W]
        num_classes: Number of segmentation classes.

    Returns:
        Mean Dice score.
    """

    predictions = torch.argmax(logits, dim=1)

    dice_scores = []

    for class_id in range(num_classes):
        prediction_class = predictions == class_id
        target_class = targets == class_id

        intersection = (
            prediction_class & target_class
        ).sum().float()

        prediction_area = prediction_class.sum().float()
        target_area = target_class.sum().float()

        denominator = prediction_area + target_area

        if denominator == 0:
            continue

        dice = (2.0 * intersection) / denominator
        dice_scores.append(dice)

    if not dice_scores:
        return 0.0

    return torch.stack(dice_scores).mean().item()