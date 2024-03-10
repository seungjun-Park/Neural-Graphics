import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple


class ComplexLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Union[torch.device, str] = None,
                 dtype: torch.dtype = torch.complex64,
                 *args,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]
        if dtype == torch.complex32:
            self.weight_dtype = torch.float16
        elif dtype == torch.complex64:
            self.weight_dtype = torch.float32
        else:
            self.weight_dtype = torch.float64

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.weight_amp = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=self.weight_dtype))
        self.weight_phase = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=self.weight_dtype))

        if bias:
            self.bias_amp = nn.Parameter(torch.empty(out_features, device=device, dtype=self.weight_dtype))
            self.bias_phase = nn.Parameter(torch.empty(out_features, device=device, dtype=self.weight_dtype))
        else:
            self.bias_amp = None
            self.bias_phase = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_amp, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_phase, a=math.sqrt(5))
        if self.bias:
            fan_in_amp, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_amp)
            fan_in_phase, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_phase)
            bound_amp = 1 / math.sqrt(fan_in_amp) if fan_in_amp > 0 else 0
            bound_phase = 1 / math.sqrt(fan_in_phase) if fan_in_phase > 0 else 0
            nn.init.uniform_(self.bias_amp, -bound_amp, bound_amp)
            nn.init.uniform_(self.bias_phase, -bound_phase, bound_phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp, phase = x.abs(), x.angle()
        amp = F.linear(amp, self.weight_amp, bias=self.bias_amp)
        phase = F.linear(phase, self.weight_phase, bias=self.bias_phase)

        x = amp * torch.exp(phase * 1j)

        return x
