import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vae.attn_block import AttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution
from modules.utils import conv_nd, group_norm, activation_func


class MiddleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 z_channels,
                 latent_dim,
                 num_attn_blocks=1,
                 dropout=0.,
                 attn_dropout=0.,
                 num_heads=-1,
                 num_head_channels=-1,
                 use_bias=True,
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

        self.middle_in = nn.ModuleList()

        layer = nn.ModuleList()

        for i in range(num_attn_blocks):
            layer.append(
                AttnBlock(
                    in_channels,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    act=act,
                )
            )

        self.middle_in.append(nn.Sequential(*layer))

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
                group_norm(in_channels),
                activation_func(act),
            )
        )

        self.quant_conv = conv_nd(dim=dim, in_channels=2 * z_channels, out_channels=2 * latent_dim, kernel_size=1)
        self.post_quant_conv = conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1)

        self.middle_out = nn.ModuleList()

        self.middle_out.append(
            nn.Sequential(
                group_norm(in_channels),
                activation_func(act),
                conv_nd(
                    dim=dim,
                    in_channels=z_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

        for i in range(num_attn_blocks):
            layer.append(
                AttnBlock(
                    in_channels,
                    heads=num_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_bias=use_bias,
                    act=act,
                )
            )

        self.middle_out.append(nn.Sequential(*layer))

    def forward(self, x):
        for module in self.middle_in:
            x = module(x)

        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)
        z = posterior.reparameterization()
        z = self.post_quant_conv(z)

        for module in self.middle_out:
            x = module(z)

        return x, posterior
