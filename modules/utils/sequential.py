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
        assert torch.is_complex(x), f'{torch.is_complex(x)}'

        x_real, x_imag = x.real, x.imag
        for module in self.modules():
            x_real = module(x_real)
            x_imag = module(x_imag)

        return torch.complex(x_real, x_imag)
