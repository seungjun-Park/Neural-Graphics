import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from utils import conv_nd, pool_nd, group_norm
from utils.checkpoints import checkpoint


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 scale_factor: Union[int, float] = 2.0,
                 dim: int = 2,
                 num_groups: int = 1,
                 pool_type: str = 'conv',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        scale_factor = int(scale_factor)
        pool_type = pool_type.lower()
        out_channels = in_channels if out_channels is None else out_channels
        if pool_type == 'conv':
            self.pooling = conv_nd(dim,
                                   in_channels,
                                   out_channels,
                                   kernel_size=scale_factor,
                                   stride=scale_factor,
                                   groups=in_channels,
                                   )

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=scale_factor, stride=scale_factor)

        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.pooling(x))
