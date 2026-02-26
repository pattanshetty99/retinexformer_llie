import torch
import torch.nn as nn

class RetinexDecomposition(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU()
        )

        self.illumination = nn.Conv2d(dim, 1, 1)
        self.reflectance = nn.Conv2d(dim, 3, 1)

    def forward(self, x):
        feat = self.encoder(x)
        I = torch.sigmoid(self.illumination(feat))
        R = torch.sigmoid(self.reflectance(feat))
        return I, R
