import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diffusion.util import (
    conv_nd,
    avg_pool_nd,
    max_pool_nd,
)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param pooling_type: a string determining if a convolution, max pooling or average pooling is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 pooling_type='conv',
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pooling_type = pooling_type.lower()
        assert self.pooling_type in ['conv', 'avg', 'max'], 'The pooling type must be either conv, avg or max.'
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if self.pooling_type == 'conv':
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        elif self.pooling_type == 'avg':
            assert self.channels == self.out_channels
            
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)