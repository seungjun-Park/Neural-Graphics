import torch
import torch.nn as nn

from utils import get_act, conv_nd
from modules.complex import ComplexLinear, CReLU, ComplexDropout, CGELU, CSiLU, ComplexConv2d


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim if embed_dim is not None else in_channels
        self.dropout = dropout

        self.fc1 = nn.Linear(self.in_channels, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.in_channels)

        self.act = get_act(act)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class ComplexMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim=None,
                 dropout=0.0,
                 bias=True,
                 act='crelu',
                 device=None,
                 dtype=torch.complex64,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        assert dtype in [torch.complex32, torch.complex64, torch.complex128]

        self.dtype = dtype
        self.in_channels = in_channels
        self.embed_dim = embed_dim if embed_dim is not None else in_channels

        self.fc1 = ComplexLinear(self.in_channels, self.embed_dim, bias=bias, device=device, dtype=dtype)
        self.fc2 = ComplexLinear(self.embed_dim, self.in_channels, bias=bias, device=device, dtype=dtype)

        self.act = CSiLU()
        self.drop = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x



