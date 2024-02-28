import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from modules.vae.attn_block import AttnBlock, FFTAttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution
from modules.utils import conv_nd, group_norm, activation_func


class MiddleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 z_channels: int,
                 latent_dim: int,
                 num_attn_blocks: int = 1,
                 dropout: int = 0.0,
                 attn_dropout: int = 0.0,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 use_bias: bool = True,
                 num_groups: int = 1,
                 act: str = 'relu',
                 dim: int = 2,
                 attn_type: str = 'vanilla',
                 **ignorekwargs
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.act = act
        self.dim = dim
        self.attn_type = attn_type.lower()

        assert self.attn_type in ['vanilla', 'fft', 'swin']

        self.middle_in = nn.ModuleList()

        for i in range(num_attn_blocks):
            if self.attn_type == 'vanilla':
                attn = AttnBlock(
                    in_channels,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    act=act,
                )
            elif self.attn_type == 'fft':
                attn = FFTAttnBlock(
                    in_channels,
                    dropout=dropout,
                    act=act,
                    fft_type='fft',
                )

            self.middle_in.append(attn)

        self.middle_in.append(
            nn.Sequential(
                conv_nd(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=z_channels * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                group_norm(in_channels, num_groups=num_groups),
                activation_func(act),
            )
        )

        self.quant_conv = conv_nd(dim=dim, in_channels=2 * z_channels, out_channels=2 * latent_dim, kernel_size=1)
        self.post_quant_conv = conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1)

        self.middle_out = nn.ModuleList()

        self.middle_out.append(
            nn.Sequential(
                conv_nd(
                    dim=dim,
                    in_channels=z_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                group_norm(in_channels, num_groups=num_groups),
                activation_func(act),
            )
        )

        for i in range(num_attn_blocks):
            if self.attn_type == 'vanilla':
                attn = AttnBlock(
                    in_channels,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    act=act,
                )
            elif self.attn_type == 'fft':
                attn = FFTAttnBlock(
                    in_channels,
                    dropout=dropout,
                    act=act,
                    fft_type='ifft',
                )

            self.middle_out.append(attn)

    def forward(self, x):
        for module in self.middle_in:
            if isinstance(module, AttnBlock) or isinstance(module, FFTAttnBlock):
                b, c, *spatial = x.shape
                x = x.reshape(b, -1, c)
                x = module(x)
                x = x.reshape(b, c, *spatial)
            else:
                x = module(x)

        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)
        x = posterior.reparameterization()
        x = self.post_quant_conv(x)

        for module in self.middle_out:
            if isinstance(module, AttnBlock) or isinstance(module, FFTAttnBlock):
                b, c, *spatial = x.shape
                x = x.reshape(b, -1, c)
                x = module(x)
                x = x.reshape(b, c, *spatial)
            else:
                x = module(x)

        return x, posterior
