import torch
import torch.nn as nn

from src.models.encoder import Encoder
from src.models.decoder import Decoder


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=7):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        bottleneck, skips = self.encoder(x)

        x = self.decoder(bottleneck, skips)

        x = self.final_conv(x)

        return x