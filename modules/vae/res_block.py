import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import activation_func, batch_norm_nd, conv_nd, group_norm


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
            activation_func(act),
            conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            group_norm(out_channels),
            activation_func(act),
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
