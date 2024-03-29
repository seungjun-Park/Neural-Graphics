import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import conv_nd
from modules.complex import ComplexConv2d


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 ):
        super().__init__()

        self.proj_out = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode='constant', value=0)
        x = self.proj_out(x)

        return x


class ComplexDownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim=2,
                 ):
        super().__init__()

        self.proj_out = ComplexConv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode='constant', value=0)
        x = self.proj_out(x)

        return x