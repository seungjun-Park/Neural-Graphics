import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable, List, Optional

from modules.utils import activation_func, group_norm, conv_nd
from modules.vae.mlp import MLP


class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 use_bias: bool = True,
                 act: str = 'gelu',
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

        self.ln = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.drop = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(in_channels,
                                               num_heads=self.heads,
                                               dropout=attn_dropout,
                                               batch_first=True,
                                               bias=use_bias)

        self.mlp = MLP(in_channels, embed_dim, dropout=dropout, act=act)
        self.ln2 = nn.LayerNorm(in_channels)

    def forward(self, x) -> torch.Tensor:
        assert len(x.shape) == 3  # x.shape == b, l, c

        h = self.ln(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.drop(h)
        h = h + x

        z = self.ln(h)
        z = self.mlp(z)

        return z + h


class FFTAttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 dropout: float = 0.,
                 act: str = 'gelu',
                 fft_type: str = 'fft',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.ln = nn.LayerNorm(in_channels)
        self.drop = nn.Dropout(dropout)

        self.fft_type = fft_type.lower()
        assert self.fft_type in ['fft', 'ifft']

        self.mlp = MLP(in_channels, embed_dim, dropout=dropout, act=act)
        self.ln2 = nn.LayerNorm(in_channels)

    def forward(self, x) -> torch.Tensor:
        assert len(x.shape) == 3  # x.shape == b, l, c

        h = self.ln(x)

        # self-attention with fft
        if self.fft_type == 'fft':
            h = torch.fft.fft(torch.fft.fft(h, dim=2), dim=1)  # c after l dimension.
        else:
            h = torch.fft.ifft(torch.fft.ifft(h, dim=1), dim=2)  # inverse order of fft attention.

        h = torch.real(h)
        h = self.drop(h)
        h = h + x

        z = self.ln(h)
        z = self.mlp(z)

        return z + h


class WindowAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 shift_size,
                 num_heads=-1,
                 num_head_channels=-1,
                 qkv_bias=True,
                 proj_bias=True,
                 attn_dropout=0.0,
                 dropout=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert num_heads != 1 or num_head_channels != -1
        if num_heads != 1:
            self.num_heads = num_heads
        elif num_head_channels != 1:
            assert in_channels % num_head_channels == 0
            self.num_heads = in_channels // num_head_channels

        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_channels, in_channels, bias=proj_bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        N = window_size[0] * window_size[1]
        relative_position_bias = self.relative_position_bias_table[relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        self.relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.shape

        pad_r = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_h, pad_w, _ = x.shape

        shift_size = self.shift_size.copy()

        if self.window_size[0] >= pad_h:
            shift_size[0] = 0
        if self.window_size[1] >= pad_w:
            shift_size[1] = 0

        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_h // self.window_size[0]) * (pad_w // self.window_size[1])
        x = x.view(b, pad_h // self.window_size[0], self.window_size[0], pad_w // self.window_size[1], self.window_size[1], c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_windows, self.window_size[0] * self.window_size[1],
                                                c)  # B*nW, Ws*Ws, C

        # multi-head attention
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias)
        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (c // self.num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = attn + self.relative_position_bias

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_h, pad_w))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -shift_size[0]), (-shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -shift_size[1]), (-shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0]: h[1], w[0]: w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_h // self.window_size[0], self.window_size[0], pad_w // self.window_size[1],
                                       self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.attention_dropout)

            x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
            x = self.proj(x)
            x = F.dropout(x, p=self.dropout)

        # reverse windows
        x = x.view(b, pad_h // self.window_size[0], pad_w // self.window_size[1], self.window_size[0], self.window_size[1], c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, : h, :w, :].contigouos()

        return x

