import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from functools import partial
from timm.models.layers import DropPath

from modules.blocks.mlp import MLP, ComplexMLP
from modules.complex import ComplexDropout, ComplexLinear, ComplexDropPath, ComplexLayerNorm, CGELU, ComplexSoftmax
from modules.blocks.utils import windows_reverse, windows_partition
from utils import to_2tuple, trunc_normal_, conv_nd, norm, group_norm


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

        self.mlp = MLP(in_channels, int(in_channels * mlp_ratio), dropout=dropout, act=act, use_conv=use_conv)

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
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 drop: float = 0.0,
                 use_conv: bool = True,
                 dim: int = 2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert num_heads != -1 or num_head_channels != -1
        if num_heads != -1:
            self.num_heads = num_heads
        elif num_head_channels != -1:
            assert in_channels % num_head_channels == 0
            self.num_heads = in_channels // num_head_channels

        self.in_channels = in_channels
        self.window_size = window_size
        head_dim = in_channels // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_conv = use_conv
        self.dim = dim

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

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

        if use_conv:
            self.qkv = nn.Conv1d(in_channels, in_channels * 3, kernel_size=1, stride=1, bias=qkv_bias)
            self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, bias=proj_bias)
        else:
            self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
            self.proj = nn.Linear(in_channels, in_channels, bias=proj_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, C, N = x.shape
        qkv = self.qkv(x).reshape(B_, self.num_heads, C // self.num_heads, N, 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
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


class ShiftedWindowAttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 window_size: int = 7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 dtype: torch.dtype = torch.complex64
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.mlp_ratio = mlp_ratio
        self.dtype = dtype

        self.w_norm1 = group_norm(in_channels)
        self.w_sma = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            proj_bias=proj_bias,
            dtype=torch.complex64,
        )

        self.w_norm2 = norm(in_channels)

        self.sw_norm1 = norm(in_channels)
        self.sw_sma = WindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            proj_bias=proj_bias,
            dtype=torch.complex64,
        )
        self.sw_norm2 = norm(in_channels)

        mlp_embed_dim = int(in_channels * mlp_ratio)
        self.w_mlp = MLP(in_channels=in_channels, embed_dim=mlp_embed_dim, dropout=drop)
        self.sw_mlp = MLP(in_channels=in_channels, embed_dim=mlp_embed_dim, dropout=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape

        if min([h, w]) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min([h, w])
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        x = x.reshape(b, -1, c)
        shortcut = x
        x = self.w_norm1(x)
        x = x.reshape(b, h, w, c)

        # partition windows
        x_windows = windows_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C
        # W-MSA
        w_attn_windows = self.w_sma(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        w_attn_windows = w_attn_windows.reshape(-1, self.window_size, self.window_size, c)

        x = windows_reverse(w_attn_windows, self.window_size, h, w)  # B H' W' C

        x = x.reshape(b, -1, c)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.w_mlp(self.w_norm2(x)))

        if self.shift_size > 0:
            x = self.sw_norm1(x)
            x = x.reshape(b, h, w, c)

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, h, w, 1)).to(device=x.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for i in h_slices:
                for j in w_slices:
                    img_mask[:, i, j, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            # cyclic shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            shifted_x_windows = windows_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            shifted_x_windows = shifted_x_windows.reshape(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            sw_attn_windows = self.sw_sma(shifted_x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            sw_attn_windows = sw_attn_windows.reshape(-1, self.window_size, self.window_size, c)

            # reverse cyclic shift

            shifted_x = windows_reverse(sw_attn_windows, self.window_size, h, w)  # B H' W' C
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            shifted_x = shifted_x.reshape(b, -1, c)
            shifted_x = x.reshape(b, -1, c) + self.drop_path(shifted_x)

            # FFN
            x = shifted_x + self.drop_path(self.sw_mlp(self.sw_norm2(shifted_x)))

        x = x.reshape(b, h, w, c)

        return x


class ComplexAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mlp_ratio: float = 4,
                 heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 bias: bool = True,
                 act: str = 'gelu',
                 use_conv=True,
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

        self.ln = nn.LayerNorm(in_channels * 2)
        self.drop = nn.Dropout(dropout)

        self.self_amp_attn = nn.MultiheadAttention(in_channels * 2,
                                                   num_heads=self.heads,
                                                   dropout=attn_dropout,
                                                   batch_first=True,
                                                   bias=bias)

        self.mlp = ComplexMLP(in_channels * 2, int(in_channels * 2 * mlp_ratio), dropout=dropout, act=act)
        self.ln2 = nn.LayerNorm(in_channels)

    def forward(self, x) -> torch.Tensor:
        assert torch.is_complex(x)
        b, c, *spatial = x.shape
        x = x.reshape(b, -1, c)

        h = self.ln(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.drop(h)
        h = h + x

        z = self.ln(h)
        z = self.mlp(z)
        z = z + h

        z = z.reshape(b, c, *spatial)

        return z


class ComplexWindowAttention(nn.Module):
    def __init__(self,
                 in_channels: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]],
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 drop: float = 0.0,
                 dtype: torch.dtype = torch.complex64,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert num_heads != -1 or num_head_channels != -1
        if num_heads != -1:
            self.num_heads = num_heads
        elif num_head_channels != -1:
            assert in_channels % num_head_channels == 0
            self.num_heads = in_channels // num_head_channels

        self.in_channels = in_channels
        self.window_size = window_size
        head_dim = in_channels // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads, dtype=dtype)
        )  # 2*Wh-1 * 2*Ww-1, nH

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

        self.qkv = ComplexLinear(in_channels, in_channels * 3, bias=qkv_bias)
        self.proj = ComplexLinear(in_channels, in_channels, bias=proj_bias)
        self.attn_drop = ComplexDropout(attn_drop)
        self.proj_drop = ComplexDropout(drop)
        self.softmax = ComplexSoftmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
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


class ComplexShiftedWindowAttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 window_size: int = 7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 dtype: torch.dtype = torch.complex64
                 ):
        super().__init__()
        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.mlp_ratio = mlp_ratio
        self.dtype = dtype

        self.w_norm1 = ComplexLayerNorm(in_channels)
        self.w_sma = ComplexWindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            proj_bias=proj_bias,
            dtype=torch.complex64,
        )

        self.w_norm2 = ComplexLayerNorm(in_channels)

        self.sw_norm1 = ComplexLayerNorm(in_channels)
        self.sw_sma = ComplexWindowAttention(
            in_channels,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            proj_bias=proj_bias,
            dtype=torch.complex64,
        )
        self.sw_norm2 = ComplexLayerNorm(in_channels)

        mlp_embed_dim = int(in_channels * mlp_ratio)
        self.w_mlp = ComplexMLP(in_channels=in_channels, embed_dim=mlp_embed_dim, dropout=drop)
        self.sw_mlp = ComplexMLP(in_channels=in_channels, embed_dim=mlp_embed_dim, dropout=drop)

        self.drop_path = ComplexDropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        b, h, w, c = x.shape

        if min([h, w]) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min([h, w])
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        x = x.reshape(b, -1, c)
        shortcut = x
        x = self.w_norm1(x)
        x = x.reshape(b, h, w, c)

        # partition windows
        x_windows = windows_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C
        # W-MSA
        w_attn_windows = self.w_sma(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        w_attn_windows = w_attn_windows.reshape(-1, self.window_size, self.window_size, c)

        x = windows_reverse(w_attn_windows, self.window_size, h, w)  # B H' W' C

        x = x.reshape(b, -1, c)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.w_mlp(self.w_norm2(x)))

        if self.shift_size > 0:
            x = self.sw_norm1(x)
            x = x.reshape(b, h, w, c)

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, h, w, 1)).to(device=x.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for i in h_slices:
                for j in w_slices:
                    img_mask[:, i, j, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            # cyclic shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            shifted_x_windows = windows_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            shifted_x_windows = shifted_x_windows.reshape(-1, self.window_size * self.window_size,
                                                          c)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            sw_attn_windows = self.sw_sma(shifted_x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            sw_attn_windows = sw_attn_windows.reshape(-1, self.window_size, self.window_size, c)

            # reverse cyclic shift

            shifted_x = windows_reverse(sw_attn_windows, self.window_size, h, w)  # B H' W' C
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            shifted_x = shifted_x.reshape(b, -1, c)
            shifted_x = x.reshape(b, -1, c) + self.drop_path(shifted_x)

            # FFN
            x = shifted_x + self.drop_path(self.sw_mlp(self.sw_norm2(shifted_x)))

        x = x.reshape(b, h, w, c)

        return x
