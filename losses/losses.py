import torch
import torch.nn as nn
from pytorch_msssim import ssim

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        return l1_loss + 0.2 * ssim_loss
