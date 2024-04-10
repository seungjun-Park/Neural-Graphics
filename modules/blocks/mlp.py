import torch
import torch.nn as nn

from utils import get_act, conv_nd


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = None,
                 dropout: float = 0.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim if embed_dim is not None else in_channels
        self.dropout = dropout

        self.use_conv = use_conv

        if use_conv:
            self.fc1 = conv_nd(dim, self.in_channels, self.embed_dim, kernel_size=1, stride=1, padding=0)
            self.fc2 = conv_nd(dim, self.embed_dim, self.in_channels, kernel_size=1, stride=1, padding=0)

        else:
            self.fc1 = nn.Linear(self.in_channels, self.embed_dim)
            self.fc2 = nn.Linear(self.embed_dim, self.in_channels)

        self.act = get_act(act)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.act(h)
        h = self.drop(h)

        return h + x




