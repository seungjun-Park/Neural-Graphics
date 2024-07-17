import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd, group_norm


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 scale_factor: Union[int, float] = 2.0,
                 dim: int = 2,
                 pool_type: str = 'conv',
                 ):
        super().__init__()

        scale_factor = int(scale_factor)

        pool_type = pool_type.lower()

        if pool_type == 'conv':
            out_channels = in_channels if out_channels is None else out_channels
            self.pooling = conv_nd(dim,
                                   in_channels,
                                   out_channels,
                                   kernel_size=scale_factor,
                                   stride=scale_factor,
                                   bias=False)

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        return x
