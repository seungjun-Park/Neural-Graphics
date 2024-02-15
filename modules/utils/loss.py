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