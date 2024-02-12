import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import conv_nd, max_pool_nd, avg_pool_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 use_conv=True
                 ):
        super().__init__()

        self.use_conv = use_conv
        if self.use_conv:
            self.layer = conv_nd(dim=dim,
                                 in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 stride=2,
                                 padding=0)
        else:
            self.layer = max_pool_nd(dim=dim,
                                     kernel_size=2,
                                     stride=2)

    def forward(self, x):
        if self.use_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)

        return self.layer(x)