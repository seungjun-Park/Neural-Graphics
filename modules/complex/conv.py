import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict
from utils import to_2tuple


class ComplexConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple] = 1,
                 padding: Union[int, List, Tuple] = 0,
                 dilation: Union[int, List, Tuple] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        self.dtype = dtype

        self.conv = nn.Conv1d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype

        return self.conv(x)


class ComplexConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple] = 1,
                 padding: Union[int, List, Tuple] = 0,
                 dilation: Union[int, List, Tuple] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=torch.complex64,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        self.dtype = dtype
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.padding = to_2tuple(padding)
        self.padding_mode = padding_mode.lower()
        self.groups = groups
        self.bias = bias
        if dtype == torch.complex32:
            self.weight_dtype = torch.float16
        elif dtype == torch.complex64:
            self.weight_dtype = torch.float32
        else:
            self.weight_dtype = torch.float64

        self.weight_amp = nn.Parameter(torch.empty(out_channels, in_channels // groups, *to_2tuple(kernel_size),
                                                   device=device, dtype=self.weight_dtype))
        self.weight_phase = nn.Parameter(torch.empty(out_channels, in_channels // groups, *to_2tuple(kernel_size),
                                                     device=device, dtype=self.weight_dtype))

        if bias:
            self.bias_amp = nn.Parameter(torch.empty(out_channels, device=device, dtype=self.weight_dtype))
            self.bias_phase = nn.Parameter(torch.empty(out_channels, device=device, dtype=self.weight_dtype))
        else:
            self.bias_amp = None
            self.bias_phase = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight_amp, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_phase, a=math.sqrt(5))
        if self.bias:
            fan_in_amp, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_amp)
            if fan_in_amp != 0:
                bound = 1 / math.sqrt(fan_in_amp)
                nn.init.uniform_(self.bias_amp, -bound, bound)

            fan_in_phase, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_phase)
            if fan_in_phase != 0:
                bound = 1 / math.sqrt(fan_in_phase)
                nn.init.uniform_(self.bias_phase, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        amp, phase = x.abs(), x.angle()

        amp = F.conv2d(amp, weight=self.weight_amp, bias=self.bias_amp, stride=self.stride, padding=self.padding, dilation=self.dilation,
                       groups=self.groups)
        phase = F.conv2d(phase, weight=self.weight_phase, bias=self.bias_phase, stride=self.stride, padding=self.padding, dilation=self.dilation,
                         groups=self.groups)

        x = amp * torch.exp(phase * 1j)

        return x
