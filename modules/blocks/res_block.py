import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath

from typing import Union, List, Tuple
from utils import get_act, conv_nd, group_norm, to_2tuple
from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock, SelfAttentionBlock


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.1,
                 drop_path: float = 0.,
                 act='relu',
                 dim=2,
                 num_groups: int = 32,
                 use_checkpoint: bool = False,
                 use_conv: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.conv1 = conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.norm1 = group_norm(in_channels, num_groups=num_groups)
        self.norm2 = group_norm(out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = conv_nd(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1)

        else:
            self.shortcut = conv_nd(dim, in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)

        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.dropout(h)

        return self.drop_path(h) + self.shortcut(x)


class AttentionResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 out_channels: int = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 num_groups: int = 32,
                 use_conv: bool = True,
                 num_heads: int = 8,
                 window_size: int = 7,
                 pretrained_window_size: int = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 attn_dropout: float = 0.,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 dim: int = 2,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.conv1 = conv_nd(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = conv_nd(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.norm1 = group_norm(in_channels, num_groups=num_groups)
        self.norm2 = group_norm(out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(dropout)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if in_res > 64:
            self.attn = DoubleWindowSelfAttentionBlock(
                in_channels=out_channels,
                in_res=to_2tuple(in_res),
                num_heads=num_heads,
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                dropout=attn_dropout,
                use_checkpoint=use_checkpoint,
                attn_mode=attn_mode,
                dim=dim
            )
        else:
            self.attn = SelfAttentionBlock(
                in_channels=out_channels,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                dropout=attn_dropout,
                use_checkpoint=use_checkpoint,
                attn_mode=attn_mode,
                dim=dim
            )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = conv_nd(dim, in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)

        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.dropout(h)

        x, attn_map = self.attn(self.shortcut(x))

        return self.drop_path(h) + x, attn_map

