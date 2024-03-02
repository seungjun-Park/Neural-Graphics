import torch
import torch.nn as nn
import torch.nn.functional as F


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
