import torch
from torch.utils.data import DataLoader
from models.retinexformer import RetinexFormer
from datasets.llie_dataset import LLIE_Dataset
from losses.losses import CombinedLoss
from utils.metrics import calculate_psnr
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RetinexFormer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = CombinedLoss()
scaler = GradScaler()

dataset = LLIE_Dataset("data/low", "data/high")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_psnr = 0

    for low, high in tqdm(loader):
        low, high = low.to(device), high.to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(low)
            loss = criterion(output, high)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_psnr += calculate_psnr(output.detach(), high)

    print(f"Epoch {epoch} PSNR: {total_psnr/len(loader)}")

    torch.save(model.state_dict(), "checkpoints/model.pth")
