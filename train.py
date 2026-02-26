import os
import torch
import random
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.retinexformer import RetinexFormer
from datasets.llie_dataset import LLIE_Dataset
from losses.losses import CombinedLoss
from utils.metrics import calculate_psnr

# ==========================
# CONFIG
# ==========================
LOW_DIR = "data/low"
HIGH_DIR = "data/high"
VAL_LOW_DIR = "data/val_low"
VAL_HIGH_DIR = "data/val_high"

BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 100
MAX_TRAIN_IMAGES = 10000
CHECKPOINT_DIR = "checkpoints"
RESUME_PATH = None  # set path if resuming

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# DATASET
# ==========================
full_dataset = LLIE_Dataset(LOW_DIR, HIGH_DIR)

if len(full_dataset) > MAX_TRAIN_IMAGES:
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    indices = indices[:MAX_TRAIN_IMAGES]
    train_dataset = Subset(full_dataset, indices)
else:
    train_dataset = full_dataset

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = LLIE_Dataset(VAL_LOW_DIR, VAL_HIGH_DIR)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ==========================
# MODEL
# ==========================
model = RetinexFormer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = CombinedLoss()
scaler = GradScaler()

start_epoch = 0
best_psnr = 0

# ==========================
# RESUME TRAINING
# ==========================
if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"] + 1
    best_psnr = checkpoint["best_psnr"]
    print(f"Resumed from epoch {start_epoch}")

# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(start_epoch, EPOCHS):

    model.train()
    train_loss = 0

    for low, high in tqdm(train_loader, desc=f"Epoch {epoch}"):

        low = low.to(device)
        high = high.to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(low)
            loss = criterion(output, high)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ==========================
    # VALIDATION
    # ==========================
    model.eval()
    total_psnr = 0

    with torch.no_grad():
        for low, high in val_loader:
            low = low.to(device)
            high = high.to(device)
            output = model(low)
            total_psnr += calculate_psnr(output, high)

    avg_psnr = total_psnr / len(val_loader)

    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val PSNR: {avg_psnr:.4f}")

    # ==========================
    # SAVE CHECKPOINT
    # ==========================
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_psnr": best_psnr
    }

    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "last.pth"))

    # Save best model
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print("Best model updated!")

print("Training Complete.")
