import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Union, List, Tuple
from utils import get_act, conv_nd, norm, group_norm


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.1,
                 act='relu',
                 dim=2,
                 groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.conv1 = conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = group_norm(out_channels, num_groups=groups)
        self.norm2 = group_norm(out_channels, num_groups=groups)
        self.dropout = nn.Dropout(dropout)
        self.act = get_act(act)

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = conv_nd(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1)

        else:
            self.shortcut = conv_nd(dim, in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)

        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = h + self.shortcut(x)

        return self.act(h)

