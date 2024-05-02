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
                 perception_level: int = 2,
                 **ignored_kwargs,
                 ):
        super().__init__()

        assert out_channels % perception_level == 0

        self.out_channels = out_channels // perception_level
        self.perception_level = perception_level

        self.conv_blocks = nn.ModuleList()

        for i in range(1, perception_level + 1):
            self.conv_blocks.append(
                conv_nd(
                    dim,
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=i,
                    dilation=i,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = list()
        for i, block in enumerate(self.conv_blocks):
            hs.append(block(x))

        return torch.cat(hs, dim=1)
