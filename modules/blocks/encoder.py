import torch

import torch.nn as nn
from typing import Tuple, List, Union

from modules.blocks.attn_block import DoubleWindowAttentionBlock
from modules.blocks.down import DownBlock
from utils import conv_nd, group_norm, to_2tuple


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 16,
                 quant_dim: int = 4,
                 logit_dim: int = 1,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_norm: bool = True,
                 ):
        super().__init__()

        self.embed = nn.Sequential(
                conv_nd(dim, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            )
        self.encoder = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            for j in range(num_blocks[i] if isinstance(num_blocks, list) else num_blocks):
                down.append(
                    DoubleWindowAttentionBlock(
                        in_channels=in_ch,
                        in_res=to_2tuple(cur_res),
                        out_channels=out_ch,
                        num_heads=num_heads,
                        window_size=window_size,
                        qkv_bias=qkv_bias,
                        proj_bias=bias,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        act=act,
                        num_groups=num_groups,
                        use_norm=use_norm,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                        use_conv=use_conv,
                        dim=dim
                    )
                )
                in_ch = out_ch

            self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, pool_type=pool_type))
                cur_res //= 2

            if use_conv:
                self.quant = conv_nd(dim, in_ch, quant_dim, kernel_size=1, stride=1)
            else:
                self.quant = nn.Linear(in_ch, quant_dim)

            self.logit_out = nn.Linear(int(quant_dim * (cur_res ** 2)), logit_dim)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = self.embed(x)

        for i, module in enumerate(self.encoder):
            h = module(h)

        h = self.quant(h)
        h = torch.flatten(h, start_dim=1)
        h = self.logit_out(h)

        return h
