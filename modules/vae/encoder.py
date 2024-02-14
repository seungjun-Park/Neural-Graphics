import torch

import torch.nn as nn
import torch.nn.functional as F

from modules.vae.down import DownBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import MHAttnBlock
from modules.utils import activation_func, conv_nd, batch_norm_nd
from modules.vae.distributions import DiagonalGaussianDistribution


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 in_resolution,
                 hidden_dim,
                 z_channels,
                 latent_dim,
                 num_res_blocks,
                 ch_mult=(1, 2, 4, 8),
                 attn_res=(1, 2, 4, 8),
                 dropout=0.,
                 resamp_with_conv=True,
                 num_heads=-1,
                 num_head_channels=-1,
                 act='relu',
                 num_classes=None,
                 dim=2,
                 **ignorekwargs
                 ):
        super().__init__()

        assert num_head_channels != -1 or num_heads != 1

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.in_resolution = in_resolution
        self.current_res = in_resolution
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.act = act
        self.dim = dim

        self.down = nn.ModuleList()

        self.conv_in = conv_nd(dim=dim,
                               in_channels=in_channels,
                               out_channels=hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        if num_classes is not None:
            self.embed_layer = nn.Sequential(
                nn.Embedding(num_classes, in_channels),
                nn.Linear(in_channels, in_channels)
            )

        in_ch = hidden_dim

        for i, mult in enumerate(ch_mult):
            layer = nn.ModuleList()
            out_ch = hidden_dim * mult

            for j in range(num_res_blocks):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            if i in attn_res:
                if self.num_heads == -1:
                    heads = in_ch // self.num_head_channels
                else:
                    heads = self.num_heads

                layer.append(MHAttnBlock(in_channels=in_ch,
                                         heads=heads))

            if i != len(ch_mult) - 1:
                layer.append(DownBlock(in_ch, dim=dim, use_conv=resamp_with_conv))
                self.current_res = self.current_res // 2

            self.down.append(nn.Sequential(*layer))

        if self.num_heads == -1:
            heads = in_ch // self.num_head_channels
        else:
            heads = self.num_heads

        self.middle_block = nn.Sequential(
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                dim=dim,
            ),
            MHAttnBlock(in_channels=in_ch,
                        heads=heads),
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                dim=dim,
            ),
        )

        self.out = nn.Sequential(
            batch_norm_nd(dim=dim, num_features=in_ch),
            activation_func(act),
            conv_nd(dim=dim,
                    in_channels=in_ch,
                    out_channels=2 * z_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)
        )

        self.quant_conv = conv_nd(dim=dim, in_channels=2 * z_channels, out_channels=2 * latent_dim, kernel_size=1)

    def forward(self, x, label=None):
        if self.num_classes is not None:
            embed = self.embed_layer(label)
            while len(embed.shape) < len(x.shape):
                embed = embed[..., None]
            x = x + embed

        x = self.conv_in(x)
        for module in self.down:
            x = module(x)
        x = self.middle_block(x)
        x = self.out(x)
        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)

        return posterior
