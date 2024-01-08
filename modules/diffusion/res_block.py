import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TimestepBlock
from .up import Upsample
from .down import Downsample
from utils.activation import Sine

from modules.diffusion.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_sine_act=False,
    ):
        super().__init__()

        act = Sine if use_sine_act else nn.SiLU

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.emb_layer = nn.Sequential(
            act(),
            linear(emb_channels, self.out_channels),
        )

        layers = [
            normalization(channels),
            act(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        ]

        if up:
            layers.append(Upsample(self.out_channels, True, dims))

        self.in_layers = nn.Sequential(*layers)

        layers = [
            normalization(self.out_channels),
            act(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        ]

        if down:
            layers.append(Downsample(self.out_channels, True, dims))

        self.out_layers = nn.Sequential(*layers)

        skip_connection = []

        if self.out_channels == channels:
            skip_connection.append(nn.Identity())
        elif use_conv:
            skip_connection.append(conv_nd(dims, channels, self.out_channels, 3, padding=1))
        else:
            skip_connection.append(conv_nd(dims, channels, self.out_channels, 1))

        if up:
            skip_connection.append(Upsample(self.out_channels, True, dims))

        if down:
            skip_connection.append(Downsample(self.out_channels, True, dims))

        self.skip_connection = nn.Sequential(*skip_connection)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layer(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h
