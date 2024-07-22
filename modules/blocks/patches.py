import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from typing import List, Union, Tuple
from utils import group_norm, conv_nd, to_2tuple, get_act
from modules.blocks.res_block import ResidualBlock


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 in_res: Union[int, List, Tuple] = 64,
                 patch_size: Union[int, List, Tuple] = 4,
                 use_conv: bool = True,
                 dim=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.in_res = in_res
        self.patch_size = patch_size
        assert self.in_res % self.patch_size == 0

        self.patch_res = in_res // patch_size
        self.num_patches = self.patch_res ** 2

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
        x = self.proj(x)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 scale_factor: int = 2,
                 num_groups: int = 1,
                 use_conv: bool = True,
                 dim: int = 2,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels * 2 if out_channels is None else out_channels
        self.scale_factor = scale_factor
        self.use_conv = use_conv

        self.reduction = conv_nd(dim, (scale_factor ** 2) * in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = self.reduction(x)
        x = self.norm(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 scale_factor: int = 2,
                 num_groups: int = 1,
                 use_conv: bool = True,
                 dim: int = 2,
                 ):
        super().__init__()

        self.scale_factor = scale_factor
        self.use_conv = use_conv
        out_channels = (in_channels if out_channels is None else out_channels)
        self.expand = conv_nd(dim, in_channels, out_channels * (self.scale_factor ** 2), kernel_size=1, stride=1)
        self.norm = group_norm(out_channels, num_groups=num_groups)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        b, c, h, w = x.shape
        x = rearrange(x, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=self.scale_factor, p2=self.scale_factor, c=c//(self.scale_factor ** 2))
        x = self.norm(x)

        return x

    