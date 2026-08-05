import torch
import torch.nn as nn

from src.models.blocks import DoubleConv


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

    def forward(self, x, skips):
        skip1, skip2, skip3, skip4 = skips

        x = self.up1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec4(x)

        return x