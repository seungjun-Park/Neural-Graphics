import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from functools import partial
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

from modules.blocks.mlp import MLP
from utils import to_2tuple, trunc_normal_, conv_nd, norm, group_norm


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
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
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H, W, -1)
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
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 drop: float = 0.0,
                 attn_mode: str = 'vanilla',
                 ):
        super().__init__()

        attn_mode = attn_mode.lower()
        assert attn_mode in ['vanilla', 'cosine']
        self.attn_mode = attn_mode

        assert num_heads != -1 or num_head_channels != -1
        if num_heads != -1:
            self.num_heads = num_heads
        elif num_head_channels != -1:
            assert in_channels % num_head_channels == 0
            self.num_heads = in_channels // num_head_channels

        self.in_channels = in_channels
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        if attn_mode == 'vanilla':
            head_dim = in_channels // self.num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, num_heads, bias=False))

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

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(in_channels))
            self.v_bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.q_bias = None
            self.v_bias = None

        self.proj = nn.Linear(in_channels, in_channels, bias=proj_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # scaled dot attention
        if self.attn_mode == 'vanilla':
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # scaled cosine attention
        else:
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
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
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 window_size: int = 7,
                 pretrained_window_size: int = 0,
                 shift_size: int = 0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act: str = 'relu',
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            proj_bias=proj_bias,
            attn_mode=attn_mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(in_channels)

        mlp_embed_dim = int(in_channels * mlp_ratio)
        self.mlp = MLP(in_channels=in_channels, embed_dim=mlp_embed_dim, dropout=drop, act=act)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.in_res
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        x = x.reshape(b, c, -1).permute(0, 2, 1) # B, C, H, W => B, N(H * W), C

        shortcut = x
        x = x.reshape(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(b, h * w, c)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        x = x.permute(0, 2, 1).reshape(b, -1, h, w)

        return x

