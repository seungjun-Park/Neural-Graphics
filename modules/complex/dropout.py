import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDropout(nn.Module):
    def __init__(self,
                 p: float = 0.5,
                 inplace: bool = False,
                 *args,
                 ):
        super().__init__()

        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep = torch.bernoulli(torch.empty(x.shape, device=x.device), p=self.p) > 0
            return torch.where(keep, x / self.p, 0)
        else:
            return x

