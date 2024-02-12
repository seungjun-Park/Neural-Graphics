import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import conv_nd


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 use_conv=True,
                 mode='nearest'):
        super().__init__()
        self.use_conv = use_conv
        self.mode = mode
        if self.use_conv:
            self.conv = conv_nd(dim=dim,
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)

        return x