import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from modules.complex import ComplexConv2d, ComplexGroupNorm, ComplexLayerNorm, ComplexLinear
from modules.blocks.patches import ComplexPatchExpanding
from modules.blocks.up import UpBlock, ComplexUpBlock
from modules.blocks.res_block import ResidualBlock, ComplexResidualBlock
from modules.blocks.attn_block import AttnBlock, FFTAttnBlock, ComplexShiftedWindowAttnBlock
from utils import get_act, conv_nd, group_norm, to_2tuple, ComplexSequential


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
        self.act = get_act(act)

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

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 1, stride=1),
                nn.Conv2d(latent_dim, in_ch, kernel_size=3, stride=1, padding=1),
            )
        )

        self.layers.append(
            nn.Sequential(
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act
                ),
                AttnBlock(
                    in_ch,
                    embed_dim=embed_dim,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    act=act,
                ),
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act
                ),
            )
        )

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
