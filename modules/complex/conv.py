import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict


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

        self.conv = nn.Conv2d(
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