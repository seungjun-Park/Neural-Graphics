import itertools
import math
import numpy as np
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional
from functools import partial
from utils.checkpoints import checkpoint

from modules.blocks.mlp import MLP, ConvMLP
from modules.blocks.positional_encoding import PositionalEncoding
from utils import to_2tuple, trunc_normal_, conv_nd, norm, group_norm, functional_conv_nd, get_act
from timm.models.layers import DropPath


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


def attention_map_window_reverse(windows, window_size, H, W):
    b, head, l, _ = windows.shape
    nH, nW = H // window_size, W // window_size
    b = int(b / (H * W / window_size / window_size))
    x = windows.reshape(b, nH, nW, head, l, _)
    x = x.reshape(b, -1, l, _)

    return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 attn_dropout: float = 0.,
                 proj_dropout: float = 0.,
                 dim: int = 2,
                 ):
        super().__init__()

        self.d_k = in_channels // num_heads
        self.num_heads = num_heads

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.scale = 1 / math.sqrt(self.d_k)

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, stride=1, bias=qkv_bias)

        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        q = self.q(x)
        if context:
            k, v = self.kv(context).chunk(2, dim=1)
        else:
            k, v = self.kv(x).chunk(2, dim=1)
        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape
        l = np.prod(spatial)

        q = q.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        k = k.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        v = v.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)

        attn = torch.einsum('bhct,bhcs->bhts', q, k)
        attn = attn * self.scale

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        v = torch.einsum('bhts,bhcs->bhct', attn, v)
        v = v.reshape(b, -1, *spatial)
        v = self.proj_dropout(self.proj(v))
        return v


class AdditiveAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 num_heads: int,
                 dropout: float = 0.1,
                 dim: int = 2,
                 ):
        super().__init__()

        self.d_k = in_channels // num_heads
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = conv_nd(dim, self.d_k * 2, self.d_k, kernel_size=1, stride=1, bias=False)
        self.pos_enc = PositionalEncoding(self.d_k, to_2tuple(in_res))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape

        q = q.reshape(b * self.num_heads, self.d_k, *spatial)
        k = k.reshape(b * self.num_heads, self.d_k, *spatial)
        v = v.reshape(b * self.num_heads, self.d_k, *spatial)

        q = self.pos_enc(q)
        k = self.pos_enc(k)

        attn_score = self.attn(torch.cat([q, k], dim=1))

        attn_score = F.sigmoid(attn_score)
        attn_score = self.dropout(attn_score)

        out = attn_score * v
        out = out.reshape(b, -1, *spatial)

        return out


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 window_size: int,
                 shift_size: int = 0,
                 pretrained_window_size: int = 0,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 qk_scale: float = None,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 attn_mode: str = 'v1',
                 dim: int = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.shift_size = shift_size
        self.d_k = in_channels // num_heads

        assert in_channels % num_heads == 0

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

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

        self.register_buffer('attn_mask', attn_mask, persistent=False)

        attn_mode = attn_mode.lower()
        assert attn_mode in ['v1', 'v2']
        self.attn_mode = attn_mode

        if attn_mode == 'v1':
            self.scale = qk_scale or self.d_k ** -0.5

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, self.num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if pretrained_window_size > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size - 1)
            else:
                relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_position_bias_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, stride=1, bias=qkv_bias)

        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        q = self.q(x)
        if context:  # cross attention
            k, v = self.kv(context).chunk(2, dim=1)
        else:
            k, v = self.kv(x).chunk(2, dim=1)

        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape
        assert spatial == self.in_res, "input feature has wrong size"

        if self.shift_size > 0:
            _, _, h, w = q.shape
            q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            k = torch.roll(k, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

        q = window_partition(q, self.window_size)  # nW*B, C, window_size, window_size
        k = window_partition(k, self.window_size)  # nW*B, C, window_size, window_size
        v = window_partition(v, self.window_size)  # nW*B, C, window_size, window_size

        l = self.window_size * self.window_size

        q = q.reshape(-1, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        k = k.reshape(-1, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        v = v.reshape(-1, c, l).reshape(b, self.num_heads, c // self.num_heads, l)

        # scaled dot attention
        if self.attn_mode == 'v1':
            attn = torch.einsum('bhct,bhcs->bhts', q, k)
            attn = attn * self.scale

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
                l, l, -1)  # Wh*Ww,Wh*Ww,nH
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
                l, l, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)

        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            attn = attn.reshape(-1, nW, self.num_heads, l, l) + self.attn_mask.unsqueeze(1).unsqueeze(0).to(attn.device)
            attn = attn.reshape(-1, self.num_heads, l, l)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        v = torch.einsum('bhts,bhcs->bhct', attn, v)
        v = v.reshape(-1, c, self.window_size, self.window_size)
        v = window_reverse(v, self.window_size, *spatial)  # B c, H' W'

        if self.shift_size > 0:
            v = torch.roll(v, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        v = self.proj_dropout(self.proj(v))

        return v


class DoubleWindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 window_size: int,
                 pretrained_window_size: int = 0,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 qk_scale: float = None,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 attn_mode: str = 'v1',
                 dim: int = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.shift_size = window_size // 2
        self.d_k = in_channels // num_heads

        assert in_channels % num_heads == 0

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

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

        self.register_buffer('attn_mask', attn_mask, persistent=False)

        attn_mode = attn_mode.lower()
        assert attn_mode in ['v1', 'v2']
        self.attn_mode = attn_mode

        if attn_mode == 'v1':
            self.scale = qk_scale or self.d_k ** -0.5

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((self.num_heads, 1, 1))), requires_grad=True)

            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, self.num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if pretrained_window_size > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size - 1)
            else:
                relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_position_bias_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, stride=1, bias=qkv_bias)

        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        q = self.q(x)
        if context:  # cross attention
            k, v = self.kv(context).chunk(2, dim=1)
        else:
            k, v = self.kv(x).chunk(2, dim=1)

        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape
        assert tuple(spatial) == self.in_res, "input feature has wrong size"

        if self.shift_size > 0:
            _, _, h, w = q.shape
            q, shifted_q = q.chunk(2, dim=1)
            k, shifted_k = k.chunk(2, dim=1)
            v, shifted_v = v.chunk(2, dim=1)

            shifted_q = torch.roll(shifted_q, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_k = torch.roll(shifted_k, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_v = torch.roll(shifted_v, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

            q = torch.cat([q, shifted_q], dim=1)
            k = torch.cat([k, shifted_k], dim=1)
            v = torch.cat([v, shifted_v], dim=1)

        q = window_partition(q, self.window_size)  # nW*B, C, window_size, window_size
        k = window_partition(k, self.window_size)  # nW*B, C, window_size, window_size
        v = window_partition(v, self.window_size)  # nW*B, C, window_size, window_size

        l = self.window_size * self.window_size

        q = q.reshape(-1, c, l).reshape(-1, self.num_heads, c // self.num_heads, l)
        k = k.reshape(-1, c, l).reshape(-1, self.num_heads, c // self.num_heads, l)
        v = v.reshape(-1, c, l).reshape(-1, self.num_heads, c // self.num_heads, l)

        # scaled dot attention
        if self.attn_mode == 'v1':
            attn = torch.einsum('bhct,bhcs->bhts', q, k)
            attn = attn * self.scale

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)].reshape(
                l, l, -1)  # Wh*Ww,Wh*Ww,nH
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
                l, l, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)

        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            attn = attn.reshape(-1, nW, self.num_heads, l, l) + self.attn_mask.unsqueeze(1).unsqueeze(0).to(
                attn.device)
            attn = attn.reshape(-1, self.num_heads, l, l)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        v = torch.einsum('bhts,bhcs->bhct', attn, v)
        v = v.reshape(-1, c, self.window_size, self.window_size)
        v = window_reverse(v, self.window_size, *spatial)  # B c, H' W'

        if self.shift_size > 0:
            v, shifted_v = v.chunk(2, dim=1)
            shifted_v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
            v = torch.cat([v, shifted_v], dim=1)

        v = self.proj_dropout(self.proj(v))

        return v


class DeformableAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: int,
                 scale_factor: int,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 qk_scale: float = None,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 act: str = 'gelu',
                 dim: int = 2,
                 ):
        super().__init__()

        assert in_channels % num_heads == 0

        self.num_heads = num_heads
        self.d_k = in_channels // num_heads
        self.scale = qk_scale if qk_scale else 1 / (self.d_k ** 2)
        self.res = in_res // scale_factor

        self.conv_offset = nn.Sequential(
            conv_nd(dim, in_channels, in_channels, kernel_size=scale_factor, stride=scale_factor),
            group_norm(in_channels),
            get_act(act),
            conv_nd(dim, in_channels, 2, kernel_size=1, stride=1, bias=False)
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.rpe_table = nn.Parameter(
            torch.zeros(self.num_heads, self.res * 2 - 1, self.res * 2 - 1)
        )
        trunc_normal_(self.rpe_table, std=0.01)

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, stride=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    @torch.no_grad()
    def _get_ref_points(self, offset: torch.Tensor):
        b, h, w, c = offset.shape
        dtype = offset.dtype
        device = offset.device

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(w - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(b, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, q: torch.Tensor):
        b, c, h, w = q.shape

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, h, dtype=q.dtype, device=q.device),
            torch.arange(0, w, dtype=q.dtype, device=q.device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(w - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(b, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        b, c, *spatial = x.shape
        l = np.prod(spatial)
        if context is None:
            context = x

        q = self.q(x)
        offset = self.conv_offset(q).contiguous()  # B * g 2 Hg Wg
        n_sample = np.prod(offset.shape[2:])

        offset = offset.permute(0, 2, 3, 1)
        reference = self._get_ref_points(offset)

        pos = (offset + reference).clamp(-1., 1.)

        sampled = F.grid_sample(
            input=context,
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        sampled = sampled.reshape(b, c, 1, n_sample)

        q = q.reshape(b * self.num_heads, self.d_k, l)
        k, v = self.kv(sampled).chunk(2, dim=1)
        k = k.reshape(b * self.num_heads, self.d_k, n_sample)
        v = v.reshape(b * self.num_heads, self.d_k, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn * self.scale

        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(b, -1, -1, -1)
        q_grid = self._get_q_grid(x)
        displacement = (q_grid.reshape(b, l, 2).unsqueeze(2) - pos.reshape(b, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(b * self.num_heads, l, n_sample)
        attn = attn + attn_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        v = torch.einsum('b m n, b c n -> b c m', attn, v)
        v = v.reshape(b, c, *spatial)
        v = self.proj_dropout(self.proj(v))

        return v


class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 window_size: int,
                 shift_size: int = 0,
                 pretrained_window_size: int = 0,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 qk_scale: float = None,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 attn_type: str = 'mha',
                 dim: int = 2,
                 scale_factor: int = 4,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        attn_type = attn_type.lower()

        if attn_type == 'mha':
            self.attn = MultiHeadAttention(
                in_channels=in_channels,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                dim=dim,
            )
        elif attn_type == 'swin-v1':
            self.attn = WindowAttention(
                in_channels=in_channels,
                in_res=to_2tuple(in_res),
                window_size=window_size,
                shift_size=shift_size,
                pretrained_window_size=pretrained_window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                attn_mode='v1',
                dim=dim,
            )

        elif attn_type == 'swin-v2':
            self.attn = WindowAttention(
                in_channels=in_channels,
                in_res=to_2tuple(in_res),
                window_size=window_size,
                shift_size=shift_size,
                pretrained_window_size=pretrained_window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                attn_mode='v2',
                dim=dim,
            )
        elif attn_type == 'dswin-v1':
            self.attn = DoubleWindowAttention(
                in_channels=in_channels,
                in_res=to_2tuple(in_res),
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                attn_mode='v1',
                dim=dim,
            )

        elif attn_type == 'dswin-v2':
            self.attn = DoubleWindowAttention(
                in_channels=in_channels,
                in_res=to_2tuple(in_res),
                window_size=window_size,
                pretrained_window_size=pretrained_window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                attn_mode='v2',
                dim=dim,
            )

        elif attn_type == 'da':
            self.attn = DeformableAttention(
                in_channels=in_channels,
                in_res=in_res,
                scale_factor=scale_factor,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_scale=qk_scale,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                dim=dim
            )

        else:
            raise NotImplementedError(f'{attn_type} was not implemented.')

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.attn(x, context)
