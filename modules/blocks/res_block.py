import torch.nn as nn

from utils import get_act, conv_nd, group_norm
from modules.complex import ComplexConv2d, ComplexBatchNorm, CSiLU, ComplexGroupNorm, ComplexDropout


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.1,
                 act='relu',
                 dim=2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim

        layer = [
            group_norm(in_channels),
            get_act(act),
            conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        ]

        self.layer = nn.Sequential(*layer)

        if self.in_channels != self.out_channels:
            self.shortcut = conv_nd(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.layer(x)
        x = self.shortcut(x)

        return x + h


class ComplexResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.0,
                 act='crelu',
                 dim=2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim

        layer = [
            ComplexGroupNorm(in_channels),
            CSiLU(),
            ComplexConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ComplexGroupNorm(out_channels),
            CSiLU(),
            ComplexDropout(dropout),
            ComplexConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        ]

        self.layer = nn.Sequential(*layer)

        if self.in_channels != self.out_channels:
            self.shortcut = ComplexConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.layer(x)

        return h + self.shortcut(x)
