import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple


class UNet(nn.Module):
    def __init__(self):
        super().__init__()




    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        return x