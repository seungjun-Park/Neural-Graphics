import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import conv_nd, pool_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int = 2,
                 use_conv: bool = False,
                 pool_type: str = 'max',
                 ):
        super().__init__()

        if use_conv:
            self.pooling = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.pooling(x)

        return x
