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
        if self.p == 0. or not self.training:
            return x

        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        keep = x.new_empty(shape).bernoulli_(keep_prob) > 0
        return torch.where(keep, x / keep_prob, 0)


class ComplexDropPath(nn.Module):
    def __init__(self,
                 p: float = 0.,
                 scale_by_keep: bool = True,
                 ):
        super().__init__()
        self.p = p
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x

        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
