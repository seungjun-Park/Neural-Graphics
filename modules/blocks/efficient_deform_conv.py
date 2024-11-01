import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

import custom_op
from utils import to_1tuple, to_2tuple, to_3tuple


class EfficientDeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 3,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 1,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 groups_off: int = 1,
                 bias_off: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.offset_field = nn.Conv2d(
            in_channels,
            in_channels * 1,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)

        return torch.ops.custom_op.efficient_deform_conv1d(
            x,
            self.weight,
            offset_field,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias
        )


class EfficientDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 3,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 1,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 groups_off: int = 1,
                 bias_off: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.offset_field = nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)

        return torch.ops.custom_op.efficient_deform_conv2d(
            x,
            self.weight,
            offset_field,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias
        )


class EfficientDeformConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 3,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 1,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 groups_off: int = 1,
                 bias_off: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.offset_field = nn.Conv2d(
            in_channels,
            in_channels * 3,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)

        return torch.ops.custom_op.efficient_deform_conv3d(
            x,
            self.weight,
            offset_field,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias
        )


def efficient_deform_conv_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return EfficientDeformConv1d(*args, **kwargs)
    elif dim == 2:
        return EfficientDeformConv2d(*args, **kwargs)
    elif dim == 3:
        return EfficientDeformConv3d(*args, **kwargs)
    else:
        raise NotImplementedError(f'{dim} is not supported.')
