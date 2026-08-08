import torch

from src.models.unet import UNet


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = UNet(
        in_channels=3,
        out_channels=7
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        3,
        512,
        512,
        device=device
    )

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    expected_shape = (2, 7, 512, 512)

    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, "
        f"got {tuple(output.shape)}"
    )

    print("Model forward-pass test PASSED.")


if __name__ == "__main__":
    main()