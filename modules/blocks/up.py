import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int = None,
                 out_channels: int = None,
                 dim: int = 2,
                 scale_factor: Union[int, float] = 2.0,
                 mode: str = 'nearest',
                 use_conv: bool = True,
                 ):
        super().__init__()
        mode = mode.lower()
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']
        self.mode = mode.lower()
        self.use_conv = use_conv
        self.scale_factor = int(scale_factor)

        if use_conv:
            assert in_channels is not None
            out_channels = in_channels if out_channels is None else out_channels

            self.conv = conv_nd(dim=dim,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        if self.use_conv:
            self.conv(x)

        return x
