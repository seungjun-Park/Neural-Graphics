import torch
from torch.fft import rfftn, ifftn


# Frequency Distance loss
def FD(target, pred, dim=2, type='l2'):
    type = type.lower()
    assert type in ['l1', 'l2']
    dim = [-i for i in range(dim, 0, -1)]
    target_freq = rfftn(target, dim=dim)
    pred_freq = rfftn(target, dim=dim)

    # general form of frequency distance
    fd = (target_freq - pred_freq).abs()
    if type == 'l2':
        fd = torch.square(fd)
    fd = torch.mean(fd, dim=[-2, -1])
    fd = torch.sum(fd, dim=[-1])
    fd = torch.sum(fd) / fd.shape[0]
    return fd


# Log Frequency Distance loss
def LFD(target, pred, dim=2, type='l1'):
    fd = FD(target, pred, dim=dim, type=type)
    lfd = torch.log(fd + 1)

    return lfd


def frequency_cosine_similarity(target, pred, dim=2, eps=1e-5):
    target_freq = img_to_freq(target, dim=dim)
    pred_freq = img_to_freq(pred, dim=dim)
    pred_freq_hermitian = pred_freq.conj().permute(0, 1, 3, 2) # conjugate transpose

    if dim == 1:
        dim = [-1]
    elif dim == 2:
        dim = [-2, -1]
    elif dim == 3:
        dim = [-3, -2, -1]
    else:
        NotImplementedError(f'dim: {dim} is not supported.')

    # calculate cos for each RGB channel
    target_freq_norm = target_freq.norm(dim=dim)
    pred_freq_hermitian_norm = pred_freq_hermitian.norm(dim=dim)

    inner_prod = torch.einsum('bcij,bcjk->bcik', target_freq, pred_freq_hermitian)
    diag = inner_prod.diagonal(offset=0, dim1=-2, dim2=-1)
    tr = diag.sum(dim=-1)

    # To prevent division by zero, we add small epsilon in denominator.
    cosine = tr / (target_freq_norm * pred_freq_hermitian_norm + eps)
    cosine = torch.sum(cosine, dim=-1)
    cosine = torch.sum(cosine) / cosine.shape[-1]

    cosine_similarity = torch.sum(1 - cosine).real

    return cosine_similarity