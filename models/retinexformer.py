import torch
import torch.nn as nn
from .decomposition import RetinexDecomposition
from .transformer_block import TransformerBlock
from .refinement import NoiseRefinement

class RetinexFormer(nn.Module):
    def __init__(self, dim=64, depth=4):
        super().__init__()

        self.decompose = RetinexDecomposition()

        self.illumination_enhancer = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1),
            *[TransformerBlock(dim) for _ in range(depth)],
            nn.Conv2d(dim, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.reflectance_refine = NoiseRefinement()

    def forward(self, x):
        I, R = self.decompose(x)
        I_enh = self.illumination_enhancer(I)
        R_ref = self.reflectance_refine(R)

        output = I_enh * R_ref
        return output.clamp(0,1)
