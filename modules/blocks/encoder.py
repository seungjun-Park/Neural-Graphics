import torch

import torch.nn as nn
from typing import Tuple, List, Union

import torchvision.models
from timm.models.layers import DropPath
from omegaconf import ListConfig

from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock, SelfAttentionBlock
from modules.blocks.patches import PatchMerging
from modules.blocks.mlp import ConvMLP, MLP
from modules.blocks.res_block import ResidualBlock
from modules.blocks.down import DownBlock
from utils import conv_nd, group_norm, to_2tuple
torchvision.models.vit_l_16()


class SwinEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 mlp_ratio: float = 4.0,
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_norm: bool = True,
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        if in_res <= 64:
            self.attn = SelfAttentionBlock(
                in_channels=in_channels,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=bias,
                dropout=attn_dropout,
                use_checkpoint=use_checkpoint,
                attn_mode=attn_mode,
                dim=dim
            )

        else:
            self.attn = DoubleWindowSelfAttentionBlock(
                in_channels=in_channels,
                in_res=to_2tuple(in_res),
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                proj_bias=bias,
                dropout=attn_dropout,
                use_checkpoint=use_checkpoint,
                attn_mode=attn_mode,
                dim=dim
            )

        self.norm1 = group_norm(in_channels, num_groups=num_groups)

        self.res_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            num_groups=num_groups,
            use_norm=use_norm,
            use_conv=use_conv,
            dim=dim,
            use_checkpoint=use_checkpoint,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.drop_path(self.attn(self.norm1(x)))
        h = self.res_block(h)
        return h


class SwinEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 16,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 num_groups: int = 16,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 mlp_ratio: float = 4.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_norm: bool = True,
                 pool_type: str = 'avg',
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.embed = nn.Sequential(
                conv_nd(dim, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            )

        self.encoder = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        self.pos_embed = nn.Parameter(torch.empty(1, embed_dim, cur_res, cur_res).normal_(std=0.02))

        if not isinstance(num_blocks, ListConfig):
            num_blocks = [num_blocks for i in range(len(hidden_dims))]
        else:
            assert len(num_blocks) == len(hidden_dims)

        if not isinstance(num_heads, ListConfig):
            num_heads = [num_heads for i in range(len(hidden_dims))]
        else:
            assert len(num_heads) == len(hidden_dims)

        for i, (out_ch, num_block, num_head) in enumerate(zip(hidden_dims, num_blocks, num_heads)):
            down = list()

            for j in range(num_block):
                down.append(
                    SwinEncoderBlock(
                        in_channels=in_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=num_head,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                        use_norm=use_norm,
                    )
                )

                in_ch = out_ch

            self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_ch,
                        dim=dim,
                        pool_type=pool_type,
                    )
                )
                cur_res //= 2

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = self.embed(x) + self.pos_embed

        for i, module in enumerate(self.encoder):
            h = module(h)

        return h
