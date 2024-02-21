import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from modules.utils import group_norm, conv_nd


class FFTAttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dropout=0.,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
        )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)
        h = torch.real(fft.fft(fft.fft(self.proj_in(x), dim=2), dim=1))
        h = self.dropout(h)
        h = x + h
        z = self.proj_out(h)
        z = z + h
        z = z.permute(0, 2, 1)

        return z.reshape(b, c, *spatial)
