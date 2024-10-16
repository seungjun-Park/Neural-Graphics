import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils import conv_nd, group_norm, conv_transpose_nd, deform_conv_nd
from utils.checkpoints import checkpoint


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 num_groups: int = 1,
                 dim: int = 2,
                 scale_factor: Union[int, float] = 2.0,
                 offset_field_channels_per_groups: int = 1,
                 mode: str = 'nearest',
                 modulation_type: str = 'none',
                 use_checkpoint: bool = True
                 ):
        super().__init__()
        mode = mode.lower()
        self.use_checkpoint = use_checkpoint
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct', 'conv']
        self.mode = mode.lower()
        self.scale_factor = int(scale_factor)

        out_channels = out_channels if out_channels is not None else in_channels

        self.up = []

        if self.mode == 'conv':
            self.up.append(
                conv_transpose_nd(
                    dim,
                    in_channels,
                    in_channels,
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                    groups=in_channels,
                )
            )

        self.up.append(
            conv_nd(
                dim,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.up = nn.Sequential(*self.up)

        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor):
        if self.mode != 'conv':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.norm(self.up(x))
