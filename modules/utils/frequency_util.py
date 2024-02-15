import torch
import torch.fft as fft


def img_to_freq(img, dim=2):
    if len(img.shape) == 4:
        _, c, _, _ = img.shape
    elif len(img.shape) == 3:
        c, _, _ = img.shape
    else:
        c = 1

    if dim == 1:
        dim = [-1]
    elif dim == 2:
        dim = [-2, -1]
    else:
        dim = [-3, -2, -1]

    # for color image
    if c == 3:
        img_r, img_g, img_b = torch.chunk(img, 3, dim=1)
        freq_r = fft.fftshift(fft.fftn(img_r, dim=dim, norm='ortho'))
        freq_g = fft.fftshift(fft.fftn(img_g, dim=dim, norm='ortho'))
        freq_b = fft.fftshift(fft.fftn(img_b, dim=dim, norm='ortho'))
        freq = torch.cat([freq_r, freq_g, freq_b], dim=1)

    # for grayscale image
    elif c == 1:
        freq = fft.fftshift(fft.fftn(img, dim=dim))

    else:
        NotImplementedError(f'color channel == {c} is not supported.')

    return freq


def freq_to_img(freq, dim=2):
    if len(freq.shape) == 4:
        _, c, _, _ = freq.shape
    elif len(freq.shape) == 3:
        c, _, _ = freq.shape
    else:
        c = 1

    if dim == 1:
        dim = [-1]
    elif dim == 2:
        dim = [-2, -1]
    else:
        dim = [-3, -2, -1]

    # for color image
    if c == 3:
        freq_r, freq_g, freq_b = torch.chunk(freq, 3, dim=1)
        img_r = fft.ifftn(fft.ifftshift(freq_r), dim=dim, norm='ortho')
        img_g = fft.ifftn(fft.ifftshift(freq_g), dim=dim, norm='ortho')
        img_b = fft.ifftn(fft.ifftshift(freq_b), dim=dim, norm='ortho')
        img = torch.cat([img_r, img_g, img_b], dim=1).abs()

    # for grayscale image
    elif c == 1:
        img = fft.ifftn(fft.ifftshift(freq), dim=dim, norm='ortho').abs()

    else:
        NotImplementedError(f'color channel == {c} is not supported.')

    return img