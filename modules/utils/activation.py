import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self,
                 eps=30.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.eps = torch.tensor(eps, requires_grad=True)

    def forward(self, x):
        return torch.sin(self.eps * x)


class Cosine(nn.Module):
    def __init__(self,
                 eps=30.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.eps = eps

    def forward(self, x):
        return torch.cos(30 * x)
