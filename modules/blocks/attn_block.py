import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.blocks.mlp import MLP, ComplexMLP
from modules.complex import ComplexDropout, ComplexLinear, ComplexDropPath
from modules.blocks.utils import windows_reverse, windows_partition, to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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
        assert len(x.shape) == 3, f'{x.shape} is not matching b, l, c'  # x.shape == b, l, c

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


class ComplexWindowAttention(nn.Module):
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
                 dtype=torch.complex64,
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
        self.attn_dropout = ComplexDropout(attn_dropout)
        self.dropout = ComplexDropout(dropout)

        self.qkv = ComplexLinear(in_channels, in_channels * 3, bias=qkv_bias)
        self.proj = ComplexLinear(in_channels, in_channels, bias=proj_bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads, dtype=dtype)
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
            attn = self.attn_dropout(attn)

            x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
            x = F.linear(x, self.proj.weight, self.proj.bias)
            x = self.dropout(x)

        # reverse windows
        x = x.view(b, pad_h // self.window_size[0], pad_w // self.window_size[1], self.window_size[0], self.window_size[1], c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, : h, :w, :].contigouos()

        return x


class ComplexShiftedWindowAttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
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
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops