import torch

import torch.nn as nn
from typing import Tuple, List, Union

from modules.complex import ComplexConv2d, ComplexBatchNorm, ComplexGroupNorm, ComplexLayerNorm, ComplexDropout, ComplexLinear
from modules.blocks.down import DownBlock, ComplexDownBlock
from modules.blocks.res_block import ResidualBlock, ComplexResidualBlock
from modules.blocks.attn_block import AttnBlock, FFTAttnBlock, ComplexShiftedWindowAttnBlock
from modules.blocks.distributions import ComplexDiagonalGaussianDistribution, DiagonalGaussianDistribution
from modules.blocks.patches import PatchMerging, ComplexPatchMerging, PatchEmbedding, ComplexPatchEmbedding
from utils import conv_nd, group_norm, to_2tuple


class EncoderBlock(nn.Module):
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
        super().__init__(*args, **kwargs)

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
                fft_type='fft',
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

    def forward(self, x) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, -1, c)
        x = self.attn_block(x)
        x = x.reshape(b, c, *spatial)

        h = self.proj_out(x)
        h = h + self.shortcut(x)
        h = self.act(self.gn(h))
        return h


class Encoder(nn.Module):
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
                 attn_type: str = 'vanilla',
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.attn_type = attn_type.lower()

        assert len(in_resolution) == 2 and len(patch_size) == 2

        self.in_res = to_2tuple(in_resolution)
        self.patch_size = to_2tuple(patch_size)
        self.patch_res = (self.in_res[0] // self.patch_size[0], self.in_res[1] // self.patch_size[1])

        assert self.patch_res[0] % (len(hidden_dims) ** 2) == 0 and self.patch_res[1] % (len(hidden_dims) ** 2) == 0

        self.num_heads = num_heads,
        self.num_head_channels = num_head_channels
        self.attn_dropout = attn_dropout
        self.use_bias = use_bias,
        self.num_groups = num_groups
        self.act = act
        self.dim = dim

        self.down = nn.ModuleList()

        self.pos_embed = nn.Parameter(torch.empty((1, embed_dim, self.patch_res[0], self.patch_res[1], )), requires_grad=True)


        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            layer = list()

            layer.append(
                EncoderBlock(
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

            layer.append(DownBlock(out_ch, dim=dim))
            self.down.append(nn.Sequential(*layer))

            in_ch = out_ch

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for module in self.down:
            x = module(x)

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
