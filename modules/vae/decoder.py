import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import MHAttnBlock
from modules.utils import activation_func, conv_nd, batch_norm_nd


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 in_resolution,
                 out_channels,
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
                 mode='nearest',
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.num_resolutions = len(ch_mult) - 1

        self.in_resolution = in_resolution

        self.current_res = in_resolution // (2 ** self.num_resolutions)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        if num_classes is not None:
            self.embed_layer = nn.Sequential(
                nn.Embedding(num_classes, latent_dim),
                nn.Linear(latent_dim, latent_dim)
            )

        in_ch = hidden_dim * ch_mult[-1]

        print(f'latent res: {self.current_res}')

        self.post_quant_conv = conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1)
        self.conv_in = conv_nd(
            dim=dim,
            in_channels=z_channels,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.num_heads == -1:
            heads = in_ch // self.num_head_channels
        else:
            heads = self.num_heads

        self.middle_block = nn.Sequential(
            ResidualBlock(in_channels=in_ch,
                          out_channels=in_ch,
                          dropout=dropout,
                          act=act,
                          dim=dim,
                          ),
            MHAttnBlock(in_channels=in_ch,
                        heads=heads),
            ResidualBlock(in_channels=in_ch,
                          out_channels=in_ch,
                          dropout=dropout,
                          act=act,
                          dim=dim,
                          ),
        )

        self.up = nn.ModuleList()

        ch_mult.reverse()

        for i, mult in enumerate(ch_mult):
            layer = nn.ModuleList()
            out_ch = hidden_dim * mult

            if i != 0:
                layer.append(UpBlock(in_ch, dim=dim, mode=mode, use_conv=resamp_with_conv))
                self.current_res = self.current_res * 2

            if i in attn_res:
                if self.num_heads == -1:
                    heads = in_ch // self.num_head_channels
                else:
                    heads = self.num_heads

                layer.append(MHAttnBlock(in_channels=in_ch,
                                         heads=heads))

            for j in range(num_res_blocks):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            self.up.append(nn.Sequential(*layer))

        self.conv_out = nn.Sequential(
            batch_norm_nd(dim=dim, num_features=in_ch),
            activation_func(act),
            conv_nd(
                dim=dim,
                in_channels=in_ch,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Hardtanh(0, 1),
        )

        assert self.in_resolution == self.current_res, f'in_res: {self.in_resolution}, cur_res: {self.current_res}'

    def forward(self, z, label=None):
        if self.num_classes is not None:
            embed = self.embed_layer(label)
            while len(embed.shape) < len(z.shape):
                embed = embed[..., None]
            z = z + embed

        z = self.post_quant_conv(z)
        z = self.conv_in(z)
        z = self.middle_block(z)
        for module in self.up:
            z = module(z)
        z = self.conv_out(z)

        return z
