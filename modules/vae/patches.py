import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Any
from modules.utils import group_norm, conv_nd, to_tuple


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 in_resolution: Union[int, List[int, int]],
                 patch_size: Union[int, List[int, int]] = 4,
                 num_groups: int = 32,
                 dim=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.in_res = to_tuple(in_resolution)
        self.patch_size = to_tuple(patch_size)
        self.patch_res = to_tuple([in_resolution[0] // patch_size[0], in_resolution[0] // patch_size[0]])
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

        self.norm = group_norm(embed_dim, num_groups=num_groups)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.in_res[0] and w == self.in_res[1]
        x = self.proj(x).flatten(2).transpose(1, 2) # b, patch_h * patch_w, embed_dim
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2) # b, patch_h * patch_w, embed_dim

        return x


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_resolution: Union[int, List[int, int]],
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_res = to_tuple(in_resolution)
        self.in_channels = in_channels
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)

    def forward(self, x):
        h, w = self.in_res
        b, l, c = x.shape

        assert l == h * w
        assert h % 2 == 0 and w % 2 == 0

        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)

        return x
    