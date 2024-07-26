import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd, group_norm, conv_transpose_nd


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dim: int = 2,
                 scale_factor: Union[int, float] = 2.0,
                 mode: str = 'nearest',
                 num_groups: int = 1,
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
            groups=in_channels,
        )

        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.norm(self.up(x))
