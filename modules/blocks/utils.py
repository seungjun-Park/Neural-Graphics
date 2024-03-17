import torch
import torch.nn as nn
import math
import warnings
from itertools import repeat
import collections.abc


def windows_partition(x: torch.Tensor, window_size: int):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, window_size, window_size)

    return windows


def windows_reverse(windows: torch.Tensor, window_size: int, h: int, w: int):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, -1, h, w)
    return x
