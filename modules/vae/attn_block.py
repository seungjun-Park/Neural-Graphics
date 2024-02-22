import math

import torch
import torch.nn as nn

from modules.utils import activation_func, group_norm, conv_nd
from typing import Any, Callable, List, Optional


class ViTBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 heads=-1,
                 num_head_channels=-1,
                 dropout=0.,
                 attn_dropout=0.,
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


        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
        )
        self.mhattn_block = nn.MultiheadAttention(in_channels, num_heads=self.heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)

        h = self.proj_in(x)
        h, _ = self.mhattn_block(h, h, h)
        h = self.dropout(h)
        h = h + x

        z = self.proj_out(h)
        z = z + h
        z = z.permute(0, 2, 1)
        z = z.reshape(b, c, *spatial)

        return z
