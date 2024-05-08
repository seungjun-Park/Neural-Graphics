import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd, group_norm


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int = None,
                 scale_factor: Union[int, float] = 2.0,
                 dim: int = 2,
                 pool_type: str = 'conv',
                 ):
        super().__init__()

        scale_factor = int(scale_factor)

        pool_type = pool_type.lower()

        if pool_type == 'conv':
            assert in_channels is not None
            self.pooling = nn.Sequential(
                conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=scale_factor,
                        stride=scale_factor,
                        bias=False,
                        groups=in_channels),
                group_norm(in_channels, in_channels)  # instance norm
            )

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        x = self.pooling(x)
        return x