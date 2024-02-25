import torch

import torch.nn as nn
import torch.nn.functional as F

from modules.vae.down import DownBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock
from modules.utils import activation_func, conv_nd, group_norm
from modules.vae.distributions import DiagonalGaussianDistribution


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 hidden_dims,
                 num_res_blocks,
                 dropout=0.,
                 resamp_with_conv=True,
                 act='relu',
                 num_classes=None,
                 dim=2,
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.act = act
        self.dim = dim

        self.down = nn.ModuleList()

        self.conv_in = conv_nd(dim=dim,
                               in_channels=in_channels,
                               out_channels=embed_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        if num_classes is not None:
            self.embed_layer = nn.Sequential(
                nn.Embedding(num_classes, in_channels),
                nn.Linear(in_channels, in_channels)
            )

        in_ch = embed_dim

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

            layer.append(DownBlock(in_ch, dim=dim, use_conv=resamp_with_conv))
            self.down.append(nn.Sequential(*layer))

    def forward(self, x, label=None):
        if self.num_classes is not None:
            embed = self.embed_layer(label)
            while len(embed.shape) < len(x.shape):
                embed = embed[..., None]
            x = x + embed

        x = self.conv_in(x)
        for module in self.down:
            x = module(x)

        return x
