import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Any


class PositionalEncoding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 use_learnable_params: bool = False,
                 ):
        super().__init__()

        if isinstance(in_res, int):
            in_res = [in_res]

        if use_learnable_params:
            self.pos_enc = nn.Parameter(torch.zeros(1, in_channels, *in_res), requires_grad=True)
        else:
            self.pos_enc = self._make_fixed_positional_encoding(in_channels, in_res)

        self.register_buffer('pos_enc', self.pos_enc)

    def _make_fixed_positional_encoding(self, in_channels: int, in_res: Union[List[int], Tuple[int]]) -> torch.Tensor:
        l = np.prod(in_res)
        encodings = torch.zeros(l, in_channels)
        position = torch.arange(0, l, dtype=torch.float32).unsqueeze(1)
        two_i = torch.arange(0, in_channels, 2, dtype=torch.float32)
        div_term = torch.exp(two_i * -(math.log(10000.0) / in_channels))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)

        return encodings.permute(1, 0).reshape(in_channels, *in_res).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_enc
