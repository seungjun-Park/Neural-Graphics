import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict


class ComplexSequential(nn.Sequential):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        assert torch.is_complex(x), f'{torch.is_complex(x)}, {x.dtype}'

        x_real, x_imag = x.real, x.imag
        for m in self:
            x_real = m(x_real)
            x_imag = m(x_imag)

        return torch.complex(x_real, x_imag)


class ConditionalSequential(nn.Sequential):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def forward(self, x, context):
        for layer in self:
            x = layer(x, context)

        return x


class AttentionSequential(nn.Sequential):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for layer in self:
            x, attn_map = layer(x)

        return x, attn_map
