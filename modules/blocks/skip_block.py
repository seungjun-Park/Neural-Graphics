import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Union, List, Tuple

from modules.blocks import DownBlock, UpBlock, ResidualBlock, WindowAttnBlock
from utils import get_act, group_norm, conv_nd, pool_nd


class ScaledSkipBlock(nn.Module):
    def __init__(self,
                 level: int,
                 skip_dims: Union[List[int], Tuple[int]],
                 unit: int,
                 dim: int = 2,
                 pool_type: str = 'max',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        self.units = unit

        for i in range(len(skip_dims) // unit):
            blocks = list()
            sd = sum(skip_dims[i * unit: (i + 1) * unit])
            scale_factor = 2 ** abs(level - i)
            if i < level:
                blocks.append(
                    pool_nd(
                        pool_type,
                        dim,
                        kernel_size=scale_factor,
                        stride=scale_factor,
                    ),
                )

            elif i > level:
                blocks.append(
                    UpBlock(
                        sd,
                        dim=dim,
                        scale_factor=scale_factor,
                        mode=mode,
                        use_conv=False,
                    ),
                )

            else:
                blocks.append(nn.Identity())

            blocks.append(
                conv_nd(
                    dim,
                    sd,
                    sd,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
            blocks.append(group_norm(sd, sd))

            self.blocks.append(nn.Sequential(*blocks))

    def forward(self, hs: Union[List[torch.Tensor], Tuple]) -> Union[List[torch.Tensor], Tuple[torch.Tensor]]:
        if self.use_checkpoint:
            return checkpoint(self._forward, hs)
        return self._forward(hs)

    def _forward(self, hs: Union[List[torch.Tensor], Tuple]) -> Union[List[torch.Tensor], Tuple[torch.Tensor]]:
        results = []
        for i, module in enumerate(self.blocks):
            results.append(module(torch.cat(hs[i * self.units: (i + 1) * self.units], dim=1)))

        return torch.cat(results, dim=1)

