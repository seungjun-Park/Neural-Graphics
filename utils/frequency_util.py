import torch
from torch.fft import rfftn, irfftn, fftshift, ifftshift


def to_freq(x, norm='backward', shift=False):
    norm = norm.lower()
    assert norm in ['forward', 'backward', 'ortho']
    assert x.ndim >= 3, f'ndim of x more than 3, the ndim of x is {x.ndim}.'

    freq = rfftn(x, norm=norm, dim=tuple(range(2, x.ndim)))

    if shift:
        freq = fftshift(freq)

    return freq


def to_x(freq, dim=2, norm='backward', shift=False):
    norm = norm.lower()
    assert norm in ['forward', 'backward', 'ortho']
    assert freq.ndim >= 3, f'ndim of freq more than 3, the ndim of freq is {freq.ndim}.'

    if shift:
        freq = ifftshift(freq)

    x = irfftn(freq, norm=norm, dim=tuple(range(2, freq.ndim)))

    return img


def freq_filter(freq, dim=2, bandwidth=[0, 1]):
    assert len(bandwidth) == 2
    if len(freq.shape) == 4:
        _, c, h, w = freq.shape
    elif len(freq.shape) == 3:
        c, h, w = freq.shape

    else:
        NotImplementedError(f'freq shape: {freq.shape} is not supported.')

    half_h, half_w = h // 2, w // 2
    eps_h = [int(half_h * bandwidth[0]), int(half_h * bandwidth[1])]
    eps_w = [int(half_w * bandwidth[0]), int(half_w * bandwidth[1])]

    filter = torch.zeros(freq.shape)
    if len(freq.shape) == 4:
        filter[:, :, half_h - eps_h[1]: half_h + eps_h[1], half_w - eps_w[1]: half_w + eps_w[1]] = 1
        filter[:, :, half_h - eps_h[0]: half_h + eps_h[0], half_w - eps_w[0]: half_w + eps_w[0]] = 0
    elif len(freq.shape) == 3:
        filter[:, half_h - eps_h[1]: half_h + eps_h[1], half_w - eps_w[1]: half_w + eps_w[1]] = 1
        filter[:, half_h - eps_h[0]: half_h + eps_h[0], half_w - eps_w[0]: half_w + eps_w[0]] = 0

    return freq * filter
