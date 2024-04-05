import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 64,
                 num_layers=3,
                 bias=False,
                 ):
        super().__init__()
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, embed_dim, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(embed_dim * nf_mult_prev, embed_dim * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=bias),
                norm_layer(embed_dim * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(embed_dim * nf_mult_prev, embed_dim * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=bias),
            norm_layer(embed_dim * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(embed_dim * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.net = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """Standard forward."""

        x = torch.cat([x, cond], dim=1)

        return self.net(x)
