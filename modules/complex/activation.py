import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)

        x_real, x_imag = x.real, x.imag
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)

        return torch.complex(x_real, x_imag)


class CLeakReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.real, x.imag
        x_real = F.leaky_relu(x_real)
        x_imag = F.leaky_relu(x_imag)

        return torch.complex(x_real, x_imag)


class CSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = x.real, x.imag
        x_real = F.silu(x_real)
        x_imag = F.silu(x_imag)

        return torch.complex(x_real, x_imag)
