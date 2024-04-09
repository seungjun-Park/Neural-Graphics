import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self,
                 net: nn.Module,
                 in_channels: int = 5,
                 embed_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 ):
        super().__init__()

        self.net = net.eval()
        self.chns = list(reversed([64, 128, 256, 512, 512]))

        self.scaling_layer = ScalingLayer()

        self.lins = nn.ModuleList()

        for i, ch in enumerate(self.chns):
            lins = list()
            in_ch = ch
            for j in range(0, i):
                lins.append(
                    nn.Dropout(dropout)
                )
                lins.append(nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, stride=2, padding=0, bias=False))
                in_ch = in_ch // 2
            lins.append(nn.Dropout(dropout))
            lins.append(nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, bias=False))
            self.lins.insert(0, nn.Sequential(*lins))

        norm_layer = nn.BatchNorm2d

        sequence = [nn.Conv2d(5, embed_dim, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(embed_dim * nf_mult_prev, embed_dim * nf_mult, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(embed_dim * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(embed_dim * nf_mult_prev, embed_dim * nf_mult, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(embed_dim * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(embed_dim * nf_mult, 1, kernel_size=3, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        feats0, feats1 = self.net(in0_input), self.net(in1_input)
        diffs = []

        for kk in range(len(self.chns)):
            diffs.append(self.lins[kk](feats0[kk] - feats1[kk]) ** 2)

        diffs = torch.cat(diffs, dim=1)
        logits = self.main(diffs)

        return logits


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, dropout: float = 0.5):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(dropout), ]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)