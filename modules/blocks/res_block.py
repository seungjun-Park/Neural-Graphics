import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.checkpoints import checkpoint
from timm.models.layers import DropPath
import math

from typing import Union, List, Tuple
from utils import get_act, conv_nd, group_norm, to_2tuple
from modules.blocks.deform_conv import deform_conv_nd


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

        self.conv1 = conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.norm1 = group_norm(out_channels, num_groups=num_groups)
        self.norm2 = group_norm(out_channels, num_groups=num_groups)
        self.norm3 = group_norm(out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)

        return self.norm3(self.drop_path(h) + self.shortcut(x))


class DeformableResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 dim: int = 2,
                 num_groups: int = 1,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
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

        deformable_groups_per_groups = 1
        if deformable_group_channels is not None:
            deformable_groups = math.gcd(in_channels // deformable_group_channels, out_channels // deformable_group_channels)
            deformable_groups_per_groups = (in_channels // deformable_group_channels) // deformable_groups

        self.block = nn.Sequential(
            deform_conv_nd(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                groups=deformable_groups,
                deformable_groups_per_groups=deformable_groups_per_groups,
                offset_scale=offset_scale,
                fix_center=fix_center
            ),
            group_norm(out_channels, num_groups=num_groups),
            get_act(act),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            deform_conv_nd(
                dim=dim,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                groups=out_channels // deformable_group_channels if deformable_group_channels is not None else deformable_groups,
                deformable_groups_per_groups=1,
                offset_scale=offset_scale,
                fix_center=fix_center,
            ),
            group_norm(out_channels, num_groups=num_groups),
            get_act(act),
            nn.Dropout(dropout)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        elif use_conv:
            self.shortcut = conv_nd(
                dim,
                in_channels,
                out_channels,
                3,
                padding=1
            )
        else:
            self.shortcut = conv_nd(dim, in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, ), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_path(self.block(x)) + self.shortcut(x)
        h = self.drop_path(self.block2(h)) + h

        return h
