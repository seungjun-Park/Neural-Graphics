import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from functools import partial
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
                 num_heads: int,
                 dropout: float = 0.1,
                 attn_mode: str = 'cosine',
                 use_checkpoint: bool = True
                 ):
        super().__init__()

        self.d_k = in_channels // num_heads
        self.num_heads = num_heads
        attn_mode = attn_mode.lower()
        assert attn_mode in ['vanilla', 'cosine']
        self.attn_mode = attn_mode
        self.use_checkpoint = use_checkpoint

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        if attn_mode == 'vanilla':
            self.scale = 1 / math.sqrt(self.d_k)
        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, q, k, v, mask)
        return self._forward(q, k, v, mask)

    def _forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape
        l = np.prod(spatial)
        q = q.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        k = k.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        v = k.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)

        if self.attn_mode == 'cosine':
            q = F.normalize(q, dim=2)
            k = F.normalize(k, dim=2)
            scale = torch.clamp(self.scale, max=torch.log(torch.tensor(1. / 0.01, device=self.scale.device))).exp()
        else:
            scale = self.scale

        attn = torch.einsum('bhct,bhcs->bhts', q, k)
        attn = attn * scale

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        attn = torch.einsum('bhts,bhcs->bhct', attn, v)
        attn = attn.reshape(b, -1, *spatial)

        return attn


class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 dropout: float = 0.,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 dim: int = 2,
                 ):
        super().__init__()

        self.attn = MultiHeadAttention(in_channels=in_channels,
                                       num_heads=num_heads,
                                       dropout=dropout,
                                       attn_mode=attn_mode,
                                       use_checkpoint=use_checkpoint
                                       )

        self.qkv = conv_nd(dim, in_channels, in_channels * 3, kernel_size=1, stride=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        attn = self.attn(q, k, v)
        attn = self.proj(attn)

        return attn


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 qkv_bias=True,
                 proj_bias=True,
                 dropout: float = 0.,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 dim: int = 2,
                 ):
        super().__init__()

        self.attn = MultiHeadAttention(in_channels=in_channels,
                                       num_heads=num_heads,
                                       dropout=dropout,
                                       attn_mode=attn_mode,
                                       use_checkpoint=use_checkpoint
                                       )

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, stride=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        kv = self.kv(context)
        k, v = torch.chunk(kv, 2, dim=1)
        attn = self.attn(q, k, v)
        attn = self.proj(attn)

        return attn


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 window_size: Union[List[int], Tuple[int]],
                 pretrained_window_size: Union[List[int], Tuple[int]] = (0, 0),
                 num_heads: int = 8,
                 qk_scale: float = None,
                 drop: float = 0.0,
                 attn_mode: str = 'vanilla',
                 use_checkpoint: bool = True
                 ):
        super().__init__()

        attn_mode = attn_mode.lower()
        assert attn_mode in ['vanilla', 'cosine']
        self.attn_mode = attn_mode

        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.d_k = in_channels // num_heads
        self.use_checkpoint = use_checkpoint

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, q, k, v, mask)
        return self._forward(q, k, v, mask)

    def _forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        assert q.shape == k.shape == v.shape
        b, c, *spatial = q.shape
        l = np.prod(spatial)
        q = q.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        k = k.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)
        v = k.reshape(b, c, l).reshape(b, self.num_heads, c // self.num_heads, l)

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


class DoubleWindowSelfAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 num_heads: int = 8,
                 window_size: int = 7,
                 pretrained_window_size: int = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 dropout: float = 0.,
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

        self.w_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            drop=dropout,
            attn_mode=attn_mode,
            use_checkpoint=use_checkpoint
        )

        self.sw_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            drop=dropout,
            attn_mode=attn_mode,
            use_checkpoint=use_checkpoint
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
        self.proj = conv_nd(dim, in_channels * (2 if self.shift_size > 0 else 1), in_channels, kernel_size=1, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        qkv = self.qkv(x)

        w_msa = self.window_attention(qkv)
        sw_msa = self.shifted_window_attention(qkv)

        if sw_msa is None:
            msa = w_msa
        else:
            msa = torch.cat([w_msa, sw_msa], dim=1)

        return self.proj(msa)

    def window_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        # partition windows
        _, _, h, w = qkv.shape
        qkv = window_partition(qkv, self.window_size)  # nW*B, C, window_size, window_size
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # W-MSA/SW-MSA
        w_msa = self.w_attn(q=q, k=k, v=v)  # nW*B, C, window_size*window_size
        w_msa = window_reverse(w_msa, self.window_size, h, w)  # B c, H' W'

        return w_msa

    def shifted_window_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        if self.shift_size > 0:
            _, _, h, w = qkv.shape
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_qkv = window_partition(shifted_qkv, self.window_size)  # nW*B, c, window_size, window_size
            q, k, v = torch.chunk(shifted_qkv, 3, dim=1)
            sw_msa = self.sw_attn(q=q, k=k, v=v, mask=self.attn_mask)
            sw_msa = window_reverse(sw_msa, self.window_size, h, w)  # B c, H' W'
            sw_msa = torch.roll(sw_msa, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

            return sw_msa

        else:
            return None


class DoubleWindowCrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 num_heads: int = 8,
                 window_size: int = 7,
                 pretrained_window_size: int = 0,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 dropout: float = 0.,
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
            drop=dropout,
            attn_mode=attn_mode,
        )

        self.sw_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            drop=dropout,
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

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels * (2 if self.shift_size > 0 else 1), in_channels, kernel_size=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, cond)
        return self._forward(x, cond)

    def _forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        q = self.q(x)
        kv = self.kv(cond)
        qkv = torch.cat([q, kv], dim=1)

        w_msa = self.window_attention(qkv)
        sw_msa = self.shifted_window_attention(qkv)

        if sw_msa is None:
            msa = w_msa
        else:
            msa = torch.cat([w_msa, sw_msa], dim=1)

        return self.proj(msa)

    def window_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        # partition windows
        _, _, h, w = qkv.shape
        qkv = window_partition(qkv, self.window_size)  # nW*B, C, window_size, window_size
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # W-MSA/SW-MSA
        w_msa = self.w_attn(q=q, k=k, v=v)  # nW*B, C, window_size*window_size
        w_msa = window_reverse(w_msa, self.window_size, h, w)  # B c, H' W'

        return w_msa

    def shifted_window_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        if self.shift_size > 0:
            _, _, h, w = qkv.shape
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_qkv = window_partition(shifted_qkv, self.window_size)  # nW*B, c, window_size, window_size
            q, k, v = torch.chunk(shifted_qkv, 3, dim=1)
            sw_msa = self.sw_attn(q=q, k=k, v=v, mask=self.attn_mask)
            sw_msa = window_reverse(sw_msa, self.window_size, h, w)  # B c, H' W'
            sw_msa = torch.roll(sw_msa, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

            return sw_msa

        else:
            return None

