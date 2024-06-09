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
            in_res = list(in_res)

        if use_learnable_params:
            self.pos_enc = nn.Parameter(torch.zeros(1, in_channels, *in_res), requires_grad=True)
        else:
            self.pos_enc = self._make_fixed_positional_encoding(in_channels, in_res)

    def _make_fixed_positional_encoding(self, in_channels: int, in_res: Union[List[int], Tuple[int]]) -> torch.Tensor:
        encodings = torch.zeros(in_channels, *in_res)
        position = torch.arange(0, np.prod(in_res), dtype=torch.float32).reshape(in_res).unsqueeze(0)
        two_i = torch.arange(0, in_channels, 2, dtype=torch.float32)
        div_term = torch.exp(two_i * -(math.log(10000.0) / in_channels))
        encodings[0::2, ...] = torch.sin(position * div_term)
        encodings[1::2, ...] = torch.cos(position * div_term)

        return encodings.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_enc
