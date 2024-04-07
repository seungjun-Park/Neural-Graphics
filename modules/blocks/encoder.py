import torch

import torch.nn as nn
from typing import Tuple, List, Union

from modules.blocks.down import DownBlock
from modules.blocks.res_block import ResidualBlock
from modules.blocks.attn_block import AttnBlock
from modules.blocks.distributions import DiagonalGaussianDistribution
from modules.blocks.patches import PatchMerging, PatchEmbedding
from utils import conv_nd, group_norm, to_2tuple


class Encoder(nn.Module):
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

        self.down = nn.ModuleList()

        self.down.append(nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
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

            if i != len(hidden_dims):
                layer.append(DownBlock(in_ch, dim=dim))
                layer.append(
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
                )

            self.down.append(nn.Sequential(*layer))

        self.out = nn.Sequential(
            group_norm(in_ch, groups),
            nn.Conv2d(in_ch, latent_dim * 2, 1),
            nn.Conv2d(latent_dim * 2, latent_dim * 2, 1)
        )

    def forward(self, x):
        for module in self.down:
            x = module(x)

        x = self.out(x)

        return x


class ComplexEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_resolution: Union[int, List[int], Tuple[int]],
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

        # assert in_resolution % window_size == 0

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
        self.layers_phase = nn.ModuleList()

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            layer = []
            for j in range(num_blocks):
                layer.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                    )
                )

                in_ch = out_ch

            if i != len(hidden_dims) - 1:
                layer.append(
                    DownBlock(in_ch)
                )

            self.layers.append(nn.Sequential(*layer))

        self.out = nn.Sequential(
                group_norm(in_ch),
                nn.Conv2d(in_ch, latent_dim * 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(latent_dim * 2, latent_dim * 2, kernel_size=1, stride=1)
            )

    def forward(self, x) -> DiagonalGaussianDistribution:
        for module in self.layers:
            x = module(x)

        x = self.out(x)

        posterior = DiagonalGaussianDistribution(x)

        return posterior
