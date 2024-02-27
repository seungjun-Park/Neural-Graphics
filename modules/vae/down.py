import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import conv_nd, max_pool_nd, avg_pool_nd


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=2,
                 ):
        super().__init__()

        self.max_pool = max_pool_nd(dim, kernel_size=2, stride=2)
        self.avg_pool = avg_pool_nd(dim, kernel_size=2, stride=2)
        self.proj_out = conv_nd(dim, in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        x_max_avg = torch.cat([x_max, x_avg], dim=1)
        x = self.proj_out(x_max_avg)

        return x