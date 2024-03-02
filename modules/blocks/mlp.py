import torch.nn as nn

from utils import activation_func


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim=None,
                 dropout=0.0,
                 act='relu',
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim if embed_dim is not None else in_channels
        self.dropout = dropout

        self.fc1 = nn.Linear(self.in_channels, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.in_channels)

        self.act = activation_func(act)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

