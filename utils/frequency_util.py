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

    return x


def freq_mask(freq, dim=2, bandwidth=(1.0, 1.0)):
    assert len(bandwidth) == dim
    if len(freq.shape) == 2 + dim:
        b, c, *spatial = freq.shape
    else:
        c, *spatial = freq.shape

    half = []
    eps = []
    for i, value in enumerate(spatial):
        half.append(value // 2)
        eps.append(int((value // 2) * bandwidth[i]))

    mask = torch.zeros(*spatial).to(freq.device)
    slice_list = []
    for i in range(len(half)):
        slice_list += [slice(half[i] - eps[i], half[i] + eps[i])]

    mask[slice_list] = 1
    while len(mask.shape) < len(freq.shape):
        mask = mask.unsqueeze(0)

    return mask

