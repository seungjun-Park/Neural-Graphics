import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd


class LargePerceptionFieldConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dim: int = 2,
                 **ignored_kwargs,
                 ):
        super().__init__()

        assert out_channels % 3 == 0

        self.out_channels = out_channels // 3

        self.conv3 = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        self.conv5 = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
        )

        self.conv7 = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = list()
        hs.append(self.conv3(x))
        hs.append(self.conv5(x))
        hs.append(self.conv7(x))

        return torch.cat(hs, dim=1)
