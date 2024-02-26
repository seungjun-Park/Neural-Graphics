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
