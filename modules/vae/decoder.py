import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock
from modules.utils import activation_func, conv_nd, group_norm


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 embed_dim,
                 hidden_dims,
                 num_res_blocks=1,
                 dropout=0.,
                 resamp_with_conv=True,
                 act='relu',
                 dim=2,
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        out_ch = hidden_dims[-1]

        self.up = nn.ModuleList()
        self.up.append(
            conv_nd(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        in_ch = out_ch
        hidden_dims.append(embed_dim)
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1: ]

        for i, out_ch in enumerate(hidden_dims):
            layer = nn.ModuleList()

            for j in range(num_res_blocks):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            layer.append(UpBlock(in_ch, dim=dim, use_conv=resamp_with_conv))

            self.up.append(nn.Sequential(*layer))

        self.up.append(
            nn.Sequential(
                group_norm(in_ch),
                activation_func(act),
                conv_nd(
                    dim=dim,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

    def forward(self, x):
        for module in self.up:
            x = module(x)

        return x
