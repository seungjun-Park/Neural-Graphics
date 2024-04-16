import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from typing import Union, List, Tuple

from modules.blocks import DownBlock, UpBlock
from utils import get_act, group_norm, conv_nd, checkpoint


class ScaledSkipBlock(nn.Module):
    def __init__(self,
                 level: int,
                 skip_dims: Union[List[int], Tuple[int]],
                 num_blocks: int,
                 act: str = 'relu',
                 num_groups: int = 32,
                 dim: int = 2,
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()

        for k in range(len(skip_dims)):
            sd = skip_dims[k]
            idx = k // (num_blocks + 1)
            if idx < level:
                self.blocks.append(
                    nn.Sequential(
                        group_norm(sd, num_groups=num_groups),
                        get_act(act),
                        DownBlock(
                            sd,
                            scale_factor=2 ** abs(level - idx),
                            dim=dim,
                            use_conv=use_conv,
                            pool_type=pool_type
                        ),
                        conv_nd(
                            dim,
                            sd,
                            sd,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        *[

                        ]
                    )
                )

            elif idx > level:
                self.blocks.append(
                    nn.Sequential(
                        group_norm(sd, num_groups=num_groups),
                        get_act(act),
                        UpBlock(
                            sd,
                            dim=dim,
                            scale_factor=2 ** abs(level - idx),
                            mode=mode,
                            use_conv=use_conv,
                        ),
                    )
                )

            else:
                self.blocks.append(nn.Identity())

    def forward(self, hs: Union[List[torch.Tensor], Tuple]) -> Union[List[torch.Tensor], Tuple[torch.Tensor]]:
        # return checkpoint(self._forward, (hs, ), self.parameters(), self.use_checkpoint)
        return self._forward(hs)

    def _forward(self, hs: Union[List[torch.Tensor], Tuple]) -> Union[List[torch.Tensor], Tuple[torch.Tensor]]:
        results = []
        for module, h in zip(self.blocks, hs):
            results.append(module(h))

        return results
