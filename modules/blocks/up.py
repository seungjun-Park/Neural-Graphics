import torch.nn as nn
import torch.nn.functional as F

from utils import conv_nd
from modules.complex import ComplexConv2d


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 mode='nearest'):
        super().__init__()
        self.mode = mode
        self.conv = conv_nd(dim=dim,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        x = self.conv(x)

        return x


class ComplexUpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 mode='nearest'):
        super().__init__()
        self.mode = mode
        self.conv = ComplexConv2d(
                                 in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        x = self.conv(x)

        return x
