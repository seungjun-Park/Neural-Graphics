import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from modules.utils import group_norm, conv_nd


class FFTAttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.norm = group_norm(in_channels)
        self.proj_out = conv_nd(1, in_channels, in_channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = torch.real(fft.fft(fft.fft(self.norm(x), dim=-1), dim=-1))
        h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)
