import os
import torch
from torch import nn
import torch.nn.functional as F

class LinearLayerWithGeGLU(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linlayer = nn.Linear(dim1, dim2)
        self.linlayer2 = nn.Linear(dim1, dim2)
        self.gelu = F.gelu()

    def forward(self, x):
        out = torch.mul(self.gelu(self.linlayer(x)),self.linlayer2(x))
