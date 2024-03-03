import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple


class ComplexLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 device=None,
                 dtype=torch.complex64,
                 *args,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lin = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)
