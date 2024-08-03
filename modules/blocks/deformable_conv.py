import numpy as np
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import to_ntuple


class DeformableConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]] = 3,
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: int = 1,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 bias: bool = True,
                 dim: int = 1,
                 ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = to_ntuple(dim)(kernel_size)

        if isinstance(stride, int):
            stride = to_ntuple(dim)(stride)

        offset_out_channels = np.prod(kernel_size)

        self.padding = padding
        self.dilation = dilation
        self.dim = dim

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * offset_out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        offset_out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2. * F.sigmoid(self.modulator_conv(x))
        if self.dim == 1:
            deformable_conv_op = torchvision.ops.deform_conv2d

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
