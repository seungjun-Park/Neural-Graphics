import torch


def windows_partition(x: torch.Tensor, window_size: int):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x. permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)

    return windows


def windows_reverse(windows: torch.Tensor, window_size: int, h: int, w: int):
    b = int(windows.shape[0] / (h * w / (window_size ** 2)))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x