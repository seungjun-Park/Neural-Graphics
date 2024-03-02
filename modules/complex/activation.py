import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert torch.is_complex(x)

        x_real, x_imag = x.real, x.imag
        x_real = F.relu(x_real)
        x_imag = F.relu(x_imag)

        return torch.complex(x_real, x_imag)
