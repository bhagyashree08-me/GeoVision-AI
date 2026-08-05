import torch
import torch.nn as nn

from src.models.blocks import DoubleConv


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

    def forward(self, x):
        skip1 = self.enc1(x)
        x = self.pool(skip1)

        skip2 = self.enc2(x)
        x = self.pool(skip2)

        skip3 = self.enc3(x)
        x = self.pool(skip3)

        skip4 = self.enc4(x)
        x = self.pool(skip4)

        bottleneck = self.bottleneck(x)

        return bottleneck, [skip1, skip2, skip3, skip4]