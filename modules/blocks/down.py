import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd, group_norm
from utils.checkpoints import checkpoint
from modules.blocks.deform_conv import deform_conv_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 num_groups: int = 1,
                 scale_factor: Union[int, float] = 2.0,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 dim: int = 2,
                 pool_type: str = 'conv',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        scale_factor = int(scale_factor)
        pool_type = pool_type.lower()

        out_channels = in_channels if out_channels is None else out_channels
        # if pool_type == 'conv':
        #     if group_channels is not None:
        #         groups = math.gcd(in_channels // group_channels, out_channels // group_channels)
        #
        #     self.pooling = conv_nd(dim,
        #                            in_channels,
        #                            out_channels,
        #                            kernel_size=scale_factor * 2 - 1,
        #                            stride=scale_factor,
        #                            padding=(scale_factor * 2 - 1) // 2,
        #                            groups=groups,
        #                            )

        if pool_type == 'conv':
            deformable_groups_per_groups = 1
            if deformable_group_channels is not None:
                deformable_groups = math.gcd(in_channels // deformable_group_channels,
                                             out_channels // deformable_group_channels)
                deformable_groups_per_groups = (in_channels // deformable_group_channels) // deformable_groups

            self.pooling = deform_conv_nd(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale_factor * 2 - 1,
                padding=(scale_factor * 2 - 1) // 2,
                stride=scale_factor,
                bias=True,
                groups=deformable_groups,
                deformable_groups_per_groups=deformable_groups_per_groups,
                offset_scale=offset_scale,
                fix_center=fix_center,
                kernel_size_off=scale_factor * 4 - 1,
                stride_off=scale_factor,
                padding_off=(scale_factor * 4 - 1) // 2
            )

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.pooling(x))

