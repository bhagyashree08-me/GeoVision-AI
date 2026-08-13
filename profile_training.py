
import time
import torch

from src.dataset.dataloader import get_dataloaders
from src.models.unet import UNet
from src.training.losses import CombinedLoss


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "/content/DeepGlobe"

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_CLASSES = 7

NUM_BATCHES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATA
# ============================================================

train_loader, _ = get_dataloaders(
    dataset_path=DATASET_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

print("Total training batches:", len(train_loader))
print("Profiling batches:", NUM_BATCHES)


# ============================================================
# MODEL
# ============================================================

model = UNet(
    in_channels=3,
    out_channels=NUM_CLASSES,
).to(device)

model.train()

criterion = CombinedLoss(
    num_classes=NUM_CLASSES
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# ============================================================
# AMP
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# PROFILING
# ============================================================

data_time = 0.0
gpu_time = 0.0

end = time.perf_counter()

for batch_idx, (images, masks) in enumerate(train_loader):

    if batch_idx >= NUM_BATCHES:
        break

    # --------------------------------------------------------
    # Data loading time
    # --------------------------------------------------------

    current = time.perf_counter()

    data_time += current - end

    # --------------------------------------------------------
    # GPU transfer
    # --------------------------------------------------------

    images = images.to(
        device,
        non_blocking=True
    )

    masks = masks.to(
        device,
        non_blocking=True
    ).long()

    if device.type == "cuda":
        torch.cuda.synchronize()

    gpu_start = time.perf_counter()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        device_type="cuda",
        enabled=use_amp
    ):

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    gpu_time += (
        time.perf_counter()
        - gpu_start
    )

    end = time.perf_counter()

    print(
        f"Batch {batch_idx + 1}/{NUM_BATCHES} | "
        f"Loss: {loss.item():.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PROFILING RESULTS")
print("=" * 60)

print(
    f"Data loading time : {data_time:.2f} sec"
)

print(
    f"GPU compute time  : {gpu_time:.2f} sec"
)

print(
    f"Data time/batch   : "
    f"{data_time / NUM_BATCHES:.3f} sec"
)

print(
    f"GPU time/batch    : "
    f"{gpu_time / NUM_BATCHES:.3f} sec"
)

print(
    f"Total measured    : "
    f"{data_time + gpu_time:.2f} sec"
)

print("=" * 60)
