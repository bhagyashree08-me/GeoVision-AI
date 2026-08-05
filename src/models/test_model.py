import torch

from src.models.unet import UNet


def main():
    model = UNet(
        in_channels=3,
        out_channels=7
    )

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        y = model(x)

    print("=" * 50)
    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)
    print("=" * 50)

    assert y.shape == (2, 7, 512, 512)

    print("✅ UNet architecture is working correctly!")


if __name__ == "__main__":
    main()