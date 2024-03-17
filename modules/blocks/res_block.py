import torch.nn as nn

from utils import get_act, conv_nd, norm, group_norm
from modules.complex import ComplexConv2d, ComplexBatchNorm, CSiLU, ComplexGroupNorm, ComplexDropout


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.1,
                 act='relu',
                 dim=2,
                 groups: int = 32,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim

        self.conv1 = conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = group_norm(in_channels, num_groups=groups)
        self.norm2 = group_norm(out_channels, num_groups=groups)
        self.dropout = nn.Dropout(dropout)
        self.act = get_act(act)

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
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = h + self.shortcut(x)

        z = self.norm2(h)
        z = self.act(z)
        z = self.conv2(z)

        z = z + h

        z = self.dropout(z)

        return z


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
