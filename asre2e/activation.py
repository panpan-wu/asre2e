import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class Swish(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):

    def __init__(self, dim: int):
        super(self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.glu(x, dim=self.dim)
