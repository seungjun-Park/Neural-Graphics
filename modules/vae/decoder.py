import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock, FFTAttnBlock
from modules.utils import activation_func, conv_nd, group_norm


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 embed_dim: int = None,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 use_bias: bool = True,
                 num_groups: int = 32,
                 dim: int = 2,
                 act: str = 'gelu',
                 attn_type: str = 'vanilla',
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.dim = dim
        self.attn_type = attn_type.lower()

        assert self.attn_type in ['vanilla', 'fft', 'swin']

        if self.attn_type == 'vanilla':
            self.attn_block = AttnBlock(
                in_channels,
                embed_dim=embed_dim,
                heads=num_heads,
                num_head_channels=num_head_channels,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_bias=use_bias,
                act=act,
            )
        elif self.attn_type == 'fft':
            self.attn_block = FFTAttnBlock(
                in_channels,
                embed_dim=embed_dim,
                dropout=dropout,
                act=act,
                fft_type='ifft',
            )
        elif self.attn_type == 'swin':
            self.attn_block = AttnBlock(
                in_channels,
                heads=num_heads,
                num_head_channels=num_head_channels,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_bias=use_bias
            )

        self.gn = group_norm(self.out_channels, num_groups=num_groups)
        self.act = activation_func(act)

        self.proj_out = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if self.in_channels != self.out_channels:
            self.shortcut = conv_nd(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)

        x = self.attn_block(x)

        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, *spatial)

        h = self.proj_out(x)
        h = h + self.shortcut(x)
        h = self.act(self.gn(h))
        return h


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: Union[List, Tuple],
                 embed_dim: int,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 use_bias: bool = True,
                 num_groups: int = 32,
                 act: str = 'relu',
                 dim: int = 2,
                 mode: str = 'nearest',
                 attn_type: str = 'vanilla',
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        in_ch = hidden_dims[-1]
        hidden_dims.append(embed_dim)
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1: ]
        self.attn_type = attn_type.lower()

        self.up = nn.ModuleList()

        for i, out_ch in enumerate(hidden_dims):
            layer = nn.ModuleList()

            layer.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    num_groups=num_groups,
                    dim=dim,
                    act=act,
                    attn_type=self.attn_type
                )
            )

            layer.append(UpBlock(out_ch, dim=dim, mode=mode))
            self.up.append(nn.Sequential(*layer))

            in_ch = out_ch

        self.up.append(
            nn.Sequential(
                group_norm(in_ch),
                activation_func(act),
                conv_nd(
                    dim=dim,
                    in_channels=in_ch,
                    out_channels=self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

    def forward(self, x):
        for module in self.up:
            x = module(x)

        return x

    def get_last_layer(self):
        return self.up[-1][-1].weight
