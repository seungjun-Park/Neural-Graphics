import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.checkpoints import checkpoint
from timm.models.layers import DropPath
import math

from typing import Union, List, Tuple
from utils import get_act, conv_nd, group_norm, to_2tuple, zero_module, freq_mask, to_ntuple


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            group_norm(in_channels, num_groups=num_groups),
            get_act(act),
            conv_nd(dim, in_channels, out_channels, 3, padding=1)
        )

        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dim, out_channels, out_channels, 3, padding=1)
            )
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Sequential(
                conv_nd(dim=dim,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1),
            )

        else:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 1),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)

        return self.drop_path(h) + self.shortcut(x)


class ResidualFFTBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            group_norm(in_channels, num_groups=num_groups),
            get_act(act),
            conv_nd(dim, in_channels, out_channels, 3, padding=1)
        )

        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            get_act(act),
            nn.Dropout(dropout),
            zero_module(
                conv_nd(dim, out_channels, out_channels, 3, padding=1)
            )
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Sequential(
                conv_nd(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=7,
                    padding=3,
                    groups=in_channels,
                    bias=False,
                ),
                group_norm(in_channels, 1),
                conv_nd(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                ),
            )

            nn.init.ones_(self.shortcut[0].weight)

        else:
            self.shortcut = nn.Sequential(
                conv_nd(dim, in_channels, out_channels, 1),
            )
            self.shortcut[0].weight.data.ones_()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)
        x_fft = torch.fft.fftshift(torch.fft.rfftn(x, dim=tuple(range(2, x.ndim)), norm='ortho'))
        mask = freq_mask(x_fft, dim=self.dim, bandwidth=to_ntuple(self.dim)(0.15))
        x_fft = x_fft * (1 - mask)
        x = torch.fft.irfftn(torch.fft.ifftshift(x_fft), dim=tuple(range(2, x.ndim)), norm='ortho')
        return self.drop_path(h) + self.shortcut(x)
