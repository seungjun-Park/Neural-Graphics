import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int = 2,
                 scale_factor: Union[int, float] = 2.0,
                 mode: str = 'nearest',
                 ):
        super().__init__()
        mode = mode.lower()
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']
        self.mode = mode.lower()
        self.scale_factor = int(scale_factor)

        self.up = conv_nd(
                dim,
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.up(x)

        return x
