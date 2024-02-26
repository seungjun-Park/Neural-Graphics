import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vae.attn_block import AttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution
from modules.utils import conv_nd, group_norm, activation_func


class MiddleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_attn_blocks=1,
                 dropout=0.,
                 attn_dropout=0.,
                 num_heads=-1,
                 num_head_channels=-1,
                 act='relu',
                 dim=2,
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.act = act
        self.dim = dim

        self.middle = nn.ModuleList()

        layer = nn.ModuleList()

        if self.num_heads == -1:
            heads = in_channels // self.num_head_channels
        else:
            heads = self.num_heads

        for j in range(num_attn_blocks):
            layer.append(
                AttnBlock(
                    in_channels=in_channels,
                    heads=heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )

        self.middle.append(nn.Sequential(*layer))

        self.middle.append(
            group_norm(in_channels),
            activation_func(act),
            conv_nd(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x):
        for module in self.middle:
            x = module(x)

        return x
