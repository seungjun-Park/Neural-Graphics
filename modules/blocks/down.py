import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int = None,
                 scale_factor: Union[int, float] = 2.0,
                 dim: int = 2,
                 use_conv: bool = False,
                 pool_type: str = 'max',
                 ):
        super().__init__()

        scale_factor = int(scale_factor)

        if use_conv:
            assert in_channels is not None
            self.pooling = conv_nd(dim, in_channels, in_channels, kernel_size=scale_factor, stride=scale_factor)

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        x = self.pooling(x)
        return x