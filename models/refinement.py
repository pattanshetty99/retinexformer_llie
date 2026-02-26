import torch
import torch.nn as nn

class NoiseRefinement(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.net(x)
