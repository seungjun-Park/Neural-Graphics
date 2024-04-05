import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.fft import rfftn, irfftn, fftshift, ifftshift
from typing import Union, List, Tuple
from utils import to_2tuple


class LearnableFourierMask(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 ):
        super().__init__()

        if isinstance(in_res, int):
            in_res = to_2tuple(in_res)
        else:
            assert len(in_res) == 2
            in_res = tuple(in_res)

        h, w = in_res[0], in_res[1]
        w = w // 2 + 1
        self.in_res = tuple([h, w])

        self.learnable_mask = nn.Parameter(torch.empty(1, in_channels, *self.in_res, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = b, c, h, w
        h = rfftn(x, dim=tuple(range(2, x.ndim)))
        h = h * F.relu(self.learnable_mask)
        h = irfftn(h, dim=tuple(range(2, h.ndim)))
        return 0.5 * (x + h)
