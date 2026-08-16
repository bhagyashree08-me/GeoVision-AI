import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):

    def __init__(
        self,
        num_classes=7,
        model_name="nvidia/mit-b0",
    ):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):

        outputs = self.model(
            pixel_values=x
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits
