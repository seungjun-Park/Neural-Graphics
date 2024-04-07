import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from modules.blocks.up import UpBlock
from modules.blocks.res_block import ResidualBlock
from modules.blocks.attn_block import AttnBlock
from utils import get_act, conv_nd, group_norm, to_2tuple


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: Union[List, Tuple],
                 embed_dim: int,
                 latent_dim: int,
                 mlp_ratio: int = 4,
                 num_blocks: int = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 groups: int = 32,
                 act: str = 'relu',
                 dim: int = 2,
                 use_conv: bool = True,
                 norm_type: str = 'group',
                 mode: str = 'nearest',
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.attn_dropout = attn_dropout
        self.bias = bias
        self.act = act
        self.dim = dim
        self.use_conv = use_conv
        self.norm_type = norm_type

        self.up = nn.ModuleList()

        self.up.append(nn.Sequential(
            group_norm(embed_dim, groups),
            nn.Conv2d(
                embed_dim,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ))

        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            layer = list()

            for j in range(num_blocks):
                layer.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        dim=dim,
                        norm_type=norm_type,
                        groups=groups,
                    )
                )
                in_ch = out_ch

            if i != 0:
                layer.append(nn.Sequential(
                    UpBlock(in_ch, dim=dim, mode=mode),
                    AttnBlock(
                        in_ch,
                        mlp_ratio=mlp_ratio,
                        heads=num_heads,
                        num_head_channels=num_head_channels,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        norm_type=norm_type,
                        groups=groups,
                    )
                ))

            self.up.insert(0, nn.Sequential(*layer))

        self.up.append(nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        ))

        self.out = nn.Sequential(
            group_norm(embed_dim, groups),
            nn.Conv2d(embed_dim, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        for module in self.up:
            x = module(x)



        return x

    def get_last_layer(self):
        return self.up[-1][-1].weight


class ComplexDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_dim: int,
                 hidden_dims: Union[List[int], Tuple[int]],
                 latent_dim: int,
                 num_blocks: int = 2,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 drop_path: float = 0.1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 proj_bias: bool = True,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_pos_enc: bool = False,
                 dtype: str = 'complex64',
                 **ignorekwargs
                 ):
        super().__init__()

        dtype = dtype.lower()
        assert dtype in ['complex32', 'complex64', 'complex128']
        if dtype == 'complex32':
            self.dtype = torch.complex32
        elif dtype == 'complex64':
            self.dtype = torch.complex64
        else:
            self.dtype = torch.complex128

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads,
        self.num_head_channels = num_head_channels
        self.attn_dropout = attn_dropout
        self.num_groups = num_groups
        self.act = act
        self.mlp_ratio = mlp_ratio
        self.use_pos_enc = use_pos_enc

        self.layers = nn.ModuleList()

        in_ch = hidden_dims[-1]
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1: ]
        hidden_dims.append(hidden_dims[-1])

        for i, out_ch in enumerate(hidden_dims):
            layer = []

            if i != 0:
                layer.append(UpBlock(in_ch))

            for j in range(num_blocks + 1):
                layer.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act
                    )
                )

                in_ch = out_ch

            self.layers.append(nn.Sequential(*layer))

        self.out = nn.Sequential(
            group_norm(in_ch),
            nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        for module in self.layers:
            x = module(x)

        x = self.out(x)

        return x

    def get_last_layer(self):
        return self.out[-1].weight
