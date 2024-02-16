import torch
from modules.utils.frequency_util import img_to_freq, freq_to_img


# Frequency Distance loss
def FD(target, pred, dim=2):
    target_freq = img_to_freq(target, dim=dim)
    pred_freq = img_to_freq(pred, dim=dim)

    fd = (target_freq - pred_freq).abs()
    fd = torch.square(fd)
    fd = torch.mean(fd, dim=[1, 2, 3])
    fd = torch.sum(fd) / fd.shape[0]

    return fd


# Log Frequency Distance loss
def LFD(target, pred, dim=2):
    fd = FD(target, pred, dim=dim)
    lfd = torch.log(fd + 1)

    return lfd

def frequency_cosine_similarity(target, pred, dim=2, eps=1e-5):
    target_freq = img_to_freq(target, dim=dim)
    pred_freq = img_to_freq(pred, dim=dim)
    pred_freq_hermitian = pred_freq.conj().permute(0, 1, 3, 2)

    target_freq_norm = target_freq.norm(dim=[-2, -1])
    pred_freq_hermitian_norm = pred_freq_hermitian.norm(dim=[-2, -1])

    inner_prod = torch.einsum('bcij,bcjk->bcik', target_freq, pred_freq_hermitian)
    diag = inner_prod.diagonal(offset=0, dim1=-2, dim2=-1)
    tr = diag.sum(dim=-1)

    cosine = tr / (target_freq_norm * pred_freq_hermitian_norm + eps)
    cosine = torch.sum(cosine, dim=-1) / cosine.shape[-1]
    cosine = torch.sum(cosine) / cosine.shape[-1]

    cosine_similarity = torch.sum(1 - cosine).real

    return cosine_similarity