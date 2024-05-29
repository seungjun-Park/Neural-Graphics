import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from timm.models.layers import DropPath
from omegaconf import ListConfig

from modules.blocks.down import DownBlock
from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock, DoubleWindowCrossAttentionBlock
from modules.blocks.mlp import MLP, ConvMLP
from modules.blocks.patches import PatchMerging
from utils import get_act, conv_nd, group_norm, to_2tuple, ConditionalSequential


class SwinDecoderBlock(nn.Module):
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

        self.cross_attn = DoubleWindowCrossAttentionBlock(
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

        if use_conv:
            self.norm1 = group_norm(in_channels, num_groups=num_groups)
            self.norm2 = group_norm(in_channels, num_groups=num_groups)
            self.norm3 = group_norm(in_channels, num_groups=num_groups)

            self.mlp = ConvMLP(
                in_channels=in_channels,
                embed_dim=int(in_channels * mlp_ratio),
                out_channels=out_channels,
                dropout=dropout,
                act=act,
                num_groups=num_groups,
                use_norm=use_norm,
                dim=dim,
                use_checkpoint=use_checkpoint,
            )

        else:
            self.norm1 = nn.LayerNorm(in_channels)
            self.norm2 = nn.LayerNorm(in_channels)
            self.norm3 = nn.LayerNorm(in_channels)

            self.mlp = MLP(
                in_channels=in_channels,
                embed_dim=int(in_channels * mlp_ratio),
                out_channels=out_channels,
                dropout=dropout,
                act=act,
                use_norm=use_norm,
                use_checkpoint=use_checkpoint,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if in_channels != out_channels:
            if use_conv:
                self.shortcut = conv_nd(dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = conv_nd(dim, in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = x + self.drop_path(self.attn(self.norm1(x)))
        h = h + self.drop_path(self.cross_attn(self.norm2(h), self.norm3(context)))
        z = self.shortcut(h) + self.drop_path(self.mlp(h))
        return z


class SwinDecoder(nn.Module):
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
                 mlp_ratio: float = 4.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_norm: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.embed = nn.Sequential(
                conv_nd(dim, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            )

        self.decoder = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                down.append(
                    SwinDecoderBlock(
                        in_channels=in_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=num_heads[i] if isinstance(num_heads, ListConfig) else num_heads,
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

            self.decoder.append(ConditionalSequential(*down))

            if i != len(hidden_dims):
                self.decoder.append(
                    PatchMerging(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        dim=dim,
                    )
                )
                cur_res //= 2
                in_ch = out_ch

        down = list()

        for j in range(num_blocks[-1] if isinstance(num_blocks, ListConfig) else num_blocks):
            down.append(
                SwinDecoderBlock(
                    in_channels=in_ch,
                    in_res=cur_res,
                    window_size=window_size,
                    num_groups=num_groups,
                    num_heads=num_heads[-1] if isinstance(num_heads, ListConfig) else num_heads,
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

        self.decoder.append(ConditionalSequential(*down))

        self.quant = nn.Sequential(
            group_norm(in_ch, num_groups=num_groups),
            get_act(act),
            conv_nd(dim, in_ch, quant_dim, kernel_size=1),
        )
        self.logit = nn.Linear(int(quant_dim * (cur_res ** 2)), logit_dim)

    def forward(self, x: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        h = self.embed(x)

        for i, module in enumerate(self.decoder):
            if isinstance(module, PatchMerging):
                h = module(h)
            else:
                h = module(h, context.pop(0))

        h = self.quant(h)
        h = torch.flatten(h, start_dim=1)
        h = self.logit(h)

        return h
