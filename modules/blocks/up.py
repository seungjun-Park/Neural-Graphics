import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd, group_norm, conv_transpose_nd, get_act
from utils.checkpoints import checkpoint
from modules.blocks.deform_conv import deform_conv_nd


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 num_groups: int = 1,
                 dim: int = 2,
                 scale_factor: Union[int, float] = 2.0,
                 act: str = 'relu',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True
                 ):
        super().__init__()
        mode = mode.lower()
        self.use_checkpoint = use_checkpoint
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']
        self.mode = mode.lower()
        self.scale_factor = int(scale_factor)

        out_channels = out_channels if out_channels is not None else in_channels
        self.norm = group_norm(in_channels, num_groups=num_groups)

        self.dw_conv3 = conv_nd(
            dim,
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )

        self.dw_conv7 = conv_nd(
            dim,
            in_channels,
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=in_channels,
        )

        self.pw_conv = conv_nd(
            dim,
            in_channels * 2,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        local_feat = self.dw_conv3(x)
        global_feat = self.dw_conv7(x)
        feats = torch.cat([local_feat, global_feat], dim=1)
        x = self.pw_conv(feats)

        return x
