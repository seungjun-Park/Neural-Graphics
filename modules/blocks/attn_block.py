import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from functools import partial
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

from modules.blocks.mlp import MLP, ConvMLP
from utils import to_2tuple, trunc_normal_, conv_nd, norm, group_norm, functional_conv_nd


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # B, H, W, C = x.shape
    # x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)

    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, window_size, window_size)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, -1, H, W)

    return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 heads: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 use_conv: bool = True,
                 ):
        super().__init__()

        self.d_k = in_channels // heads
        self.heads = heads
        self.use_conv = use_conv

        if use_conv:
            self.q = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=bias)
            self.k = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=bias)
            self.v = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.q = nn.Linear(in_features=in_channels, out_features=in_channels, bias=bias)
            self.k = nn.Linear(in_features=in_channels, out_features=in_channels, bias=bias)
            self.v = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, c, l = q.shape

        if self.use_conv:
            q = self.q(q).reshape(b * self.heads, self.d_k, l)
            k = self.k(k).reshape(b * self.heads, self.d_k, l)
            v = self.v(v).reshape(b * self.heads, self.d_k, l)

            scores = torch.einsum('bct,bcs->bts', q, k)
            scores *= self.scale
            scores = self.softmax(scores)
            scores = self.dropout(scores)

            scores = torch.einsum('bts,bcs->bct', scores, v)
            scores = scores.reshape(b, -1, l)

        else:
            q = q.permute(0, 2, 1).contiguous()
            k = k.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()

            q = self.q(q).reshape(b, l, self.heads, self.d_k)
            k = self.k(k).reshape(b, l, self.heads, self.d_k)
            v = self.v(v).reshape(b, l, self.heads, self.d_k)

            scores = torch.einsum('bihd,bjhd -> bijh', q, k)
            scores *= self.scale

            scores = self.softmax(scores)
            scores = self.dropout(scores)

            scores = torch.einsum("bijh,bjhd->bihd", scores, v)
            scores = scores.reshape(b, l, -1).permute(0, 2, 1)  # b, l, c -> b, c, l

        return scores


class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mlp_ratio: int = 4,
                 heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 bias: bool = True,
                 act: str = 'gelu',
                 use_conv: bool = True,
                 dim: int = 2,
                 groups: int = 32,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        assert heads != -1 or num_head_channels != -1
        if heads != -1:
            assert in_channels % heads == 0
            self.heads = heads
        else:
            assert in_channels % num_head_channels == 0
            self.heads = in_channels // num_head_channels

        self.use_conv = use_conv
        self.dim = dim

        self.drop = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(in_channels=in_channels,
                                       heads=self.heads,
                                       dropout=attn_dropout,
                                       bias=bias,
                                       use_conv=use_conv)

        self.mlp = MLP(in_channels, int(in_channels * mlp_ratio), dropout=dropout, act=act)

        self.norm1 = group_norm(in_channels, groups)
        self.norm2 = group_norm(in_channels, groups)

    def forward(self, x) -> torch.Tensor:
        # In general, the shape of x is b, c, *_. denote that len(*_) == dim.
        b, c, *_ = x.shape
        x = x.reshape(b, c, -1)
        h = x

        h = self.norm1(h)
        h = self.attn(h, h, h)
        h = self.drop(h)
        h = h + x

        z = self.norm2(h)
        z = self.mlp(z)
        z = z + h

        z = z.reshape(b, -1, *_)

        return z


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 window_size: Union[List[int], Tuple[int]],
                 pretrained_window_size: Union[List[int], Tuple[int]] = (0, 0),
                 num_heads: int = 8,
                 qk_scale: float = None,
                 drop: float = 0.0,
                 attn_mode: str = 'vanilla',
                 ):
        super().__init__()

        attn_mode = attn_mode.lower()
        assert attn_mode in ['vanilla', 'cosine']
        self.attn_mode = attn_mode

        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.d_k = in_channels // num_heads

        self.in_channels = in_channels
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        if attn_mode == 'vanilla':
            self.scale = qk_scale or self.d_k ** -0.5

            self.relative_position_bias = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, self.num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            else:
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_position_bias_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.drop = nn.Dropout(drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, c, *spatial = qkv.shape
        l = np.prod(spatial)
        qkv = qkv.reshape(b, c, l)
        qkv = qkv.reshape(b, self.num_heads, c // self.num_heads, -1)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # scaled dot attention
        if self.attn_mode == 'vanilla':
            attn = torch.einsum('bhct,bhcs->bhts', q, k)
            attn = attn * self.scale

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # scaled cosine attention
        else:
            q = F.normalize(q, dim=2)
            k = F.normalize(k, dim=2)
            attn = torch.einsum('bhct,bhcs->bhts', q, k)

            scale = torch.clamp(self.scale, max=torch.log(torch.tensor(1. / 0.01, device=self.scale.device))).exp()
            attn = attn * scale

            relative_position_bias_table = self.cpb_mlp(self.relative_position_bias_table).reshape(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(b // nW, nW, self.num_heads, l, l) + mask.unsqueeze(1).unsqueeze(0).to(attn.device)
            attn = attn.reshape(-1, self.num_heads, l, l)

        attn = F.softmax(attn, dim=1)
        attn = self.drop(attn)

        attn = torch.einsum('bhts,bhcs->bhct', attn, v)
        attn = attn.reshape(b, -1, *spatial)

        return attn


class WindowAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return


class DoubleWindowAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 out_channels: int = None,
                 mlp_ratio: float = 4.0,
                 num_heads: int = 8,
                 window_size: int = 7,
                 pretrained_window_size: int = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 num_groups: int = 1,
                 use_norm: bool = True,
                 use_conv: bool = True,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 dim: int = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = self.window_size // 2
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        assert in_channels % num_heads == 0

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm = group_norm(in_channels, num_groups=num_heads)
        self.w_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            drop=attn_dropout,
            attn_mode=attn_mode,
        )

        self.sw_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            drop=attn_dropout,
            attn_mode=attn_mode,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.in_res
            img_mask = torch.zeros((1, 1, H, W))  # 1 1 H W
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, 1, window_size, window_size
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.attn_mask = attn_mask

        self.qkv = conv_nd(dim, in_channels, in_channels * 3, kernel_size=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=proj_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
            )
            if in_channels != out_channels:
                self.shortcut = conv_nd(dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1)

            else:
                self.shortcut = nn.Identity()

        else:
            self.norm1 = nn.LayerNorm(in_channels)
            self.norm2 = nn.LayerNorm(out_channels)

            self.mlp = MLP(
                in_channels=in_channels,
                embed_dim=int(in_channels * mlp_ratio),
                out_channels=out_channels,
                dropout=dropout,
                act=act,
                use_norm=use_norm
            )
            if in_channels != out_channels:
                self.shortcut = nn.Linear(in_channels, out_channels)

            else:
                self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        shortcut = x
        qkv = self.qkv(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
        else:
            shifted_qkv = x

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, C, window_size, window_size
        shifted_qkv = window_partition(shifted_qkv, self.window_size)  # nW*B, c, window_size, window_size

        # W-MSA/SW-MSA
        w_msa = self.w_attn(qkv)  # nW*B, C, window_size*window_size
        sw_msa = self.sw_attn(shifted_qkv, mask=self.attn_mask)
        # merge windows

        w_msa = window_reverse(w_msa, self.window_size, h, w)  # B c, H' W'
        sw_msa = window_reverse(sw_msa, self.window_size, h, w)  # B c, H' W'

        # reverse cyclic shift
        if self.shift_size > 0:
            sw_msa = torch.roll(sw_msa, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        msa = torch.cat([w_msa, sw_msa], dim=1)
        msa = self.proj(msa)

        msa = shortcut + self.drop_path(self.norm1(msa))
        msa = self.shortcut(msa) + self.norm2(self.mlp(msa))

        return msa

