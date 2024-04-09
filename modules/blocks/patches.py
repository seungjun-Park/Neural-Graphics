import torch
import torch.nn as nn
from einops import rearrange

from typing import List, Union, Tuple
from utils import group_norm, conv_nd, to_2tuple
from modules.complex import ComplexLayerNorm, ComplexLinear, ComplexGroupNorm, ComplexConv2d


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 in_resolution: Union[int, List, Tuple] = (64, 64),
                 patch_size: Union[int, List, Tuple] = (4, 4),
                 dim=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.in_res = to_2tuple(in_resolution)
        self.patch_size = to_2tuple(patch_size)
        assert self.in_res[0] % self.patch_size[0] == 0 and self.in_res[1] % self.patch_size[1] == 0

        self.patch_res = to_2tuple([self.in_res[0] // self.patch_size[0], self.in_res[1] // self.patch_size[1]])
        self.num_patches = self.patch_res[0] * self.patch_res[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.norm = group_norm(embed_dim, num_groups=1)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.in_res[0] and w == self.in_res[1]
        x = self.proj(x)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 scale_factor: int = 2,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_res = to_2tuple(in_resolution)
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if out_channels is None else out_channels
        self.scale_factor = scale_factor

        self.norm = nn.LayerNorm((scale_factor ** 2) * in_channels)
        self.reduction = nn.Linear((scale_factor ** 2) * in_channels, out_channels, bias=False)

    def forward(self, x):
        x = rearrange(x, 'b c (h s_h) (w s_w) -> b (c s_h s_w) h w', s_w=self.scale_factor, s_h=self.scale_factor)
        b, c, h, w = x.shape

        x = x.reshape(b, c, -1).permute(0, 2, 1)

        x = self.norm(x)
        x = self.reduction(x)

        x = x.permute(0, 2, 1).reshape(b, -1, h, w)

        return x


class PatchExpanding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.norm = nn.LayerNorm(in_channels)
        self.expand = nn.Linear(in_channels, (scale_factor ** 2) * out_channels, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, w)
        x = rearrange(x, 'b (s_h s_w c) h w -> b c (h s_h) (w s_w)', s_h=self.scale_factor, s_w=self.scale_factor)

        return x


class ComplexPatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 in_resolution: Union[int, List, Tuple] = (64, 64),
                 patch_size: Union[int, List, Tuple] = (4, 4),
                 num_groups: int = 32,
                 dim=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.in_res = to_2tuple(in_resolution)
        self.patch_size = to_2tuple(patch_size)
        assert self.in_res[0] % self.patch_size[0] == 0 and self.in_res[1] % self.patch_size[1] == 0

        self.patch_res = to_2tuple([self.in_res[0] // self.patch_size[0], self.in_res[0] // self.patch_size[0]])
        self.num_patches = self.patch_res[0] * self.patch_res[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = ComplexConv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.norm = ComplexGroupNorm(embed_dim, groups=num_groups)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.in_res[0] and w == self.in_res[1]
        x = self.proj(x)
        x = self.norm(x)

        return x


class ComplexPatchMerging(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 dtype: torch.dtype = torch.complex64,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.norm = ComplexLayerNorm((scale_factor ** 2) * in_channels)
        self.reduction = ComplexLinear((scale_factor ** 2) * in_channels, out_channels, bias=False, dtype=dtype)

        self.dtype = dtype

    def forward(self, x):
        x = rearrange(x, 'b (h s_h) (w s_w) c -> b h w (c s_h s_w)', s_w=self.scale_factor, s_h=self.scale_factor)
        b, h, w, c = x.shape

        x = x.reshape(b, -1, c)

        x = self.norm(x)
        x = self.reduction(x)

        x = x.reshape(b, h, w, -1)

        return x


class ComplexPatchExpanding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 dtype: torch.dtype = torch.complex64,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.scale_factor = scale_factor

        self.norm = ComplexLayerNorm(in_channels, dtype=dtype)
        self.expand = ComplexLinear(in_channels, (scale_factor ** 2) * in_channels, bias=False, dtype=dtype)
        self.lin = ComplexLinear(in_channels, out_channels, bias=False, dtype=dtype)

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)
        x = self.norm(x)
        x = self.expand(x)
        x = x.reshape(b, h, w, -1)
        x = rearrange(x, 'b h w (s_h s_w c) -> b (h s_h) (w s_w) c', s_h=self.scale_factor, s_w=self.scale_factor)
        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)
        x = self.lin(x)
        x = x.reshape(b, h, w, -1)

        return x

    