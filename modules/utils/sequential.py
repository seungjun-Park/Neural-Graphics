import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.vae.encoder import EncoderBlock
from modules.vae.decoder import DecoderBlock


class EmbedSequential(nn.Sequential):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for module in self.modules():
            if isinstance(module, EncoderBlock) or isinstance(module, DecoderBlock):
                b, c, *spatial = x.shape
                x = x.reshape(b, -1, c)
                x = x.permute(0, 2, 1)
                x = module(x)
                x = x.permute(0, 2, 1)
                x = x.reshape(b, c, *spatial)

            else:
                x = module(x)

        return x
