import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from functools import partial
from utils.checkpoints import checkpoint

from modules.blocks.mlp import MLP, ConvMLP
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

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if attn_mode == 'vanilla':
            self.scale = 1 / math.sqrt(self.d_k)
        else:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return checkpoint(self._forward, (q, k, v, mask), self.parameters(), self.use_checkpoint)

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

        score = torch.einsum('bhts,bhcs->bhct', attn, v)
        score = score.reshape(b, -1, *spatial)

        return score, attn


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
        self.spatial_proj = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=proj_bias, groups=in_channels)
        self.depth_proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        attn, attn_map = self.attn(q, k, v)
        attn = self.spatial_proj(attn)
        attn = self.depth_proj(attn)

        return attn, attn_map


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
        self.spatial_proj = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=proj_bias,
                                    groups=in_channels)
        self.depth_proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        kv = self.kv(context)
        k, v = torch.chunk(kv, 2, dim=1)
        attn, attn_map = self.attn(q, k, v)
        attn = self.spatial_proj(attn)
        attn = self.depth_proj(attn)

        return attn, attn_map


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 window_size: Union[List[int], Tuple[int]],
                 pretrained_window_size: Union[List[int], Tuple[int]] = (0, 0),
                 num_heads: int = 8,
                 qk_scale: float = None,
                 dropout: float = 0.0,
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

            self.relative_position_bias_table = nn.Parameter(
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

        self.dropout = nn.Dropout(dropout)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        score = torch.einsum('bhts,bhcs->bhct', attn, v)
        score = score.reshape(b, -1, *spatial)

        return score, attn


class WindowSelfAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 num_heads: int = 8,
                 window_size: int = 7,
                 shift_size: int = 0,
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
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        assert in_channels % num_heads == 0

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            dropout=dropout,
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
        self.spatial_proj = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=proj_bias,
                                    groups=in_channels)
        self.depth_proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        qkv = self.qkv(x)

        if self.shift_size > 0:
            _, _, h, w = qkv.shape
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_qkv = window_partition(shifted_qkv, self.window_size)  # nW*B, c, window_size, window_size
            q, k, v = torch.chunk(shifted_qkv, 3, dim=1)
            msa, attn_map = self.attn(q=q, k=k, v=v, mask=self.attn_mask)
            msa = window_reverse(msa, self.window_size, h, w)  # B c, H' W'
            msa = torch.roll(msa, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        else:
            # partition windows
            _, _, h, w = qkv.shape
            qkv = window_partition(qkv, self.window_size)  # nW*B, C, window_size, window_size
            q, k, v = torch.chunk(qkv, 3, dim=1)

            # W-MSA/SW-MSA
            msa, attn_map = self.attn(q=q, k=k, v=v)  # nW*B, C, window_size*window_size
            msa = window_reverse(msa, self.window_size, h, w)  # B c, H' W'

        msa = self.spatial_proj(msa)
        msa = self.depth_proj(msa)

        return msa, attn_map


class WindowCrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[List[int], Tuple[int]],
                 num_heads: int = 8,
                 window_size: int = 7,
                 shift_size: int = 0,
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
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        assert in_channels % num_heads == 0

        if min(self.in_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.in_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            dropout=dropout,
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

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        q = self.q(x)
        kv = self.kv(context)
        qkv = torch.cat([q, kv], dim=1)

        if self.shift_size > 0:
            _, _, h, w = qkv.shape
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
            shifted_qkv = window_partition(shifted_qkv, self.window_size)  # nW*B, c, window_size, window_size
            q, k, v = torch.chunk(shifted_qkv, 3, dim=1)
            msa, attn_map = self.attn(q=q, k=k, v=v, mask=self.attn_mask)
            msa = window_reverse(msa, self.window_size, h, w)  # B c, H' W'
            msa = torch.roll(msa, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        else:
            # partition windows
            _, _, h, w = qkv.shape
            qkv = window_partition(qkv, self.window_size)  # nW*B, C, window_size, window_size
            q, k, v = torch.chunk(qkv, 3, dim=1)

            # W-MSA/SW-MSA
            msa, attn_map = self.attn(q=q, k=k, v=v)  # nW*B, C, window_size*window_size
            msa = window_reverse(msa, self.window_size, h, w)  # B c, H' W'

        msa = self.proj(msa)

        return msa, attn_map


class DoubleWindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 8,
                 window_size: int = 7,
                 shift_size: int = 0,
                 pretrained_window_size: int = 0,
                 qk_scale=None,
                 dropout: float = 0.,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla',
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.w_attn = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            pretrained_window_size=to_2tuple(self.pretrained_window_size),
            num_heads=num_heads,
            qk_scale=qk_scale,
            dropout=dropout,
            attn_mode=attn_mode,
            use_checkpoint=use_checkpoint
        )

        if self.shift_size > 0:
            self.sw_attn = WindowAttention(
                in_channels,
                window_size=to_2tuple(self.window_size),
                pretrained_window_size=to_2tuple(self.pretrained_window_size),
                num_heads=num_heads,
                qk_scale=qk_scale,
                dropout=dropout,
                attn_mode=attn_mode,
                use_checkpoint=use_checkpoint
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # partition windows
        _, _, h, w = q.shape
        win_q = window_partition(q, self.window_size)  # nW*B, C, window_size, window_size
        win_k = window_partition(k, self.window_size)
        win_v = window_partition(v, self.window_size)

        # W-MSA/SW-MSA
        msa, attn_map = self.w_attn(q=win_q, k=win_k, v=win_v)  # nW*B, C, window_size*window_size
        msa = window_reverse(msa, self.window_size, h, w)  # B c, H' W'
        if self.shift_size > 0:
            shifted_win_q = window_partition(torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1)), self.window_size)
            shifted_win_k = window_partition(torch.roll(k, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1)), self.window_size)
            shifted_win_v = window_partition(torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1)), self.window_size)

            sw_msa, sw_attn_map = self.sw_attn(q=shifted_win_q, k=shifted_win_k, v=shifted_win_v, mask=mask)
            sw_msa = torch.roll(window_reverse(sw_msa, self.window_size, h, w), shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

            msa = torch.cat([msa, sw_msa], dim=1)
            attn_map = torch.cat([attn_map, sw_attn_map], dim=1)

        return msa, attn_map


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

        self.attn = DoubleWindowAttention(
            in_channels,
            window_size=self.window_size,
            shift_size=self.shift_size,
            pretrained_window_size=self.pretrained_window_size,
            num_heads=num_heads,
            qk_scale=qk_scale,
            dropout=dropout,
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
        self.proj = conv_nd(dim, in_channels * 2 if self.shift_size > 0 else in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> Tuple:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> Tuple:
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        msa, attn_map = self.attn(q=q, k=k, v=v, mask=self.attn_mask)
        msa = self.proj(msa)

        return msa, attn_map


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

        self.attn = DoubleWindowAttention(
            in_channels,
            window_size=self.window_size,
            shift_size=self.shift_size,
            pretrained_window_size=self.pretrained_window_size,
            num_heads=num_heads,
            qk_scale=qk_scale,
            dropout=dropout,
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

        self.q = conv_nd(dim, in_channels, in_channels, kernel_size=1, bias=qkv_bias)
        self.kv = conv_nd(dim, in_channels, in_channels * 2, kernel_size=1, bias=qkv_bias)
        self.proj = conv_nd(dim, in_channels * 2 if self.shift_size > 0 else in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple:
        return checkpoint(self._forward, (x, context), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple:
        H, W = self.in_res
        b, c, h, w = x.shape
        assert h * w == H * W, "input feature has wrong size"

        q = self.q(x)
        kv = self.kv(context)
        qkv = torch.cat([q, kv], dim=1)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        msa, attn_map = self.attn(q=q, k=k, v=v, mask=self.attn_mask)
        msa = self.proj(msa)

        return msa, attn_map
