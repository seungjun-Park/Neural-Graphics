import torch

import torch.nn as nn
import torch.nn.functional as F

from modules.vae.down import DownBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock
from modules.utils import activation_func, conv_nd, group_norm
from modules.vae.distributions import DiagonalGaussianDistribution


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 max_seq_len=5000,
                 num_heads=-1,
                 num_head_channels=-1,
                 dropuout=0.0,
                 attn_dropout=0.0,
                 use_bias=True,
                 num_group=32,
                 dim=2,
                 act='gelu',
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dropout = dropuout
        self.attn_dropuout = attn_dropout
        self.dim = dim
        self.num_group = num_group

        self.attn_block = AttnBlock(
            in_channels,
            max_seq_len=max_seq_len,
            heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropuout,
            attn_dropout=attn_dropout,
            bias=use_bias
        )

        self.conv = conv_nd(
            dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
        )

        self.norm = group_norm(in_channels, num_groups=num_group)
        self.act = activation_func(act)

        if self.in_channels != self.out_channels:
            self.shortcut = conv_nd(dim=dim,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.attn_block(x)
        z = F.dropout(self.conv(h), self.dropout)
        z = z + self.shortcut(h)
        z = self.act(self.norm(z))

        return z


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 hidden_dims,
                 image_size=[64, 64],
                 num_heads=-1,
                 num_head_channels=-1,
                 dropout=0.,
                 attn_dropout=0.0,
                 use_bias=True,
                 num_group=32,
                 act='relu',
                 dim=2,
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        assert len(image_size) == 2

        self.image_size = image_size
        self.num_heads = num_heads,
        self.num_head_channels = num_head_channels
        self.attn_dropout = attn_dropout
        self.use_bias = use_bias,
        self.num_group = num_group
        self.act = act
        self.dim = dim

        self.down = nn.ModuleList()

        self.conv_in = conv_nd(dim=dim,
                               in_channels=in_channels,
                               out_channels=embed_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            layer = nn.ModuleList()

            layer.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    max_seq_len=image_size[0] * image_size[1],
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropuout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    num_group=num_group,
                    dim=dim,
                    act=act,
                )
            )

            layer.append(DownBlock(out_ch, dim=dim))

            image_size = [image_size[0] // 2, image_size[1] // 2]

            self.down.append(nn.Sequential(*layer))

            in_ch = out_ch

    def forward(self, x):
        x = self.conv_in(x)
        for module in self.down:
            x = module(x)

        return x
