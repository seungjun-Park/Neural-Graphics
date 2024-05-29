import torch

import torch.nn as nn
from typing import Tuple, List, Union
from timm.models.layers import DropPath
from omegaconf import ListConfig

from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock
from modules.blocks.patches import PatchMerging
from modules.blocks.mlp import ConvMLP, MLP
from utils import conv_nd, group_norm, to_2tuple


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

        if use_conv:
            self.norm1 = group_norm(in_channels, num_groups=num_groups)
            self.norm2 = group_norm(out_channels, num_groups=num_groups)

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
            self.norm2 = nn.LayerNorm(out_channels)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.drop_path(self.attn(self.norm1(x)))
        z = self.shortcut(h) + self.drop_path(self.mlp(h))
        return z


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

        self.encoder = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                down.append(
                    SwinEncoderBlock(
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

            self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims):
                self.encoder.append(
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
                SwinEncoderBlock(
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

        self.encoder.append(nn.Sequential(*down))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        hs = []
        h = self.embed(x)

        for i, module in enumerate(self.encoder):
            h = module(h)
            if not isinstance(module, PatchMerging):
                hs.append(h)

        return hs
