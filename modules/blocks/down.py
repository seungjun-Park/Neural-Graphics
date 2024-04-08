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

        self.use_conv = use_conv

        if use_conv:
            self.pooling = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=2, padding=0)

        else:
            self.pooling = pool_nd(pool_type, dim=dim)

    def forward(self, x):
        if self.use_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.pooling(x)

        else:
            x = self.pooling(x)

        return x
