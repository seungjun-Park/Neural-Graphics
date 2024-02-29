import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock, FFTAttnBlock
from modules.utils import activation_func, conv_nd, group_norm, to_tuple, ComplexSequential


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
        x = x.reshape(b, -1, c)
        x = self.attn_block(x)
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
                 in_resolution: Union[List, Tuple] = (64, 64),
                 patch_size: Union[List, Tuple] = (4, 4),
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

        assert len(in_resolution) == 2 and len(patch_size) == 2

        self.in_res = to_tuple(in_resolution)
        self.patch_size = to_tuple(patch_size)
        self.patch_res = (self.in_res[0] // self.patch_size[0], self.in_res[1] // self.patch_size[1])

        assert self.patch_res[0] % (len(hidden_dims) ** 2) == 0 and self.patch_res[1] % (len(hidden_dims) ** 2) == 0

        in_ch = hidden_dims[-1]
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1:]
        hidden_dims.append(embed_dim)

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

        for i in range(patch_size[0] // 2):
            self.up.append(
                nn.Sequential(
                    ResidualBlock(
                        in_channels=in_ch,
                        dropout=dropout,
                        act=act,
                        dim=dim,
                    ),
                    UpBlock(in_ch, dim=dim, mode=mode)
                )
            )

        self.up.append(
            nn.Sequential(
                group_norm(in_ch),
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


class FDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: Union[List, Tuple],
                 embed_dim: int,
                 z_channels: int,
                 latent_dim: int,
                 num_res_blocks: int = 2,
                 attn_res: Union[List, Tuple] = (),
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
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1:]
        hidden_dims.append(embed_dim)

        self.attn_type = attn_type.lower()

        self.up = nn.ModuleList()

        self.up.append(
            ComplexSequential(
                conv_nd(
                    dim,
                    in_channels=latent_dim,
                    out_channels=z_channels,
                    kernel_size=1
                ),
                conv_nd(
                    dim,
                    in_channels=z_channels,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

        self.up.append(
            ComplexSequential(
                ResidualBlock(
                    in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act,
                    dim=dim,
                ),
                AttnBlock(
                    in_ch,
                    embed_dim=embed_dim,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    act=act,
                ),
                ResidualBlock(
                    in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act,
                    dim=dim,
                )
            )
        )

        for i, out_ch in enumerate(hidden_dims):
            layer = nn.ModuleList()

            layer.append(UpBlock(in_ch, dim=dim, mode=mode))

            for j in range(num_res_blocks):
                layer.append(
                    ResidualBlock(
                        in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        dim=dim,
                    )
                )
                in_ch = out_ch

            if i in attn_res:
                layer.append(
                    AttnBlock(
                        in_ch,
                        embed_dim=embed_dim,
                        heads=num_heads,
                        num_head_channels=num_head_channels,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        use_bias=use_bias,
                        act=act,
                    )
                )

            self.up.append(ComplexSequential(*layer))

        self.up.append(
            ComplexSequential(
                group_norm(in_ch, num_groups=num_groups),
                conv_nd(
                    dim,
                    in_channels=in_ch,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

    def forward(self, x):
        for module in self.up:
            x = module(x)

        return x

    def get_last_layer(self):
        return self.up[-1][-1].weight
