import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd, group_norm


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dim: int = 2,
                 num_groups: int = 32,
                 scale_factor: Union[int, float] = 2.0,
                 mode: str = 'nearest',
                 ):
        super().__init__()
        mode = mode.lower()
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']
        self.mode = mode.lower()
        self.scale_factor = int(scale_factor)

        out_channels = out_channels if out_channels is not None else in_channels

        self.up = conv_nd(
                dim,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.up(x)
        x = self.norm(x)
        return x
