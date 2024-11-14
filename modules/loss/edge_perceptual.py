import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight, freq_mask
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss, wasserstein_d_loss, LFD
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 lpips_weight: float = 1.0,
                 bdcn_weight: float = 1.0,
                 freq_weight: float = 1.0,
                 ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.lpips_weight = lpips_weight
        self.bdcn_weight = bdcn_weight
        self.freq_weight = freq_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split="train"):
        preds = preds.repeat(1, 3, 1, 1).contiguous()
        labels = labels.repeat(1, 3, 1, 1).contiguous()

        lpips_loss = self.perceptual_loss(preds, labels).mean()
        bdcn_loss = bdcn_loss2(1 - preds, 1 - labels).mean()

        freq = torch.fft.fftshift(torch.fft.rfftn(imgs, dim=tuple(range(2, imgs.ndim)), norm='ortho'))
        mask = freq_mask(freq, dim=2, bandwidth=[0.2, 0.2])
        freq *= (1 - mask)
        freq = torch.fft.irfftn(torch.fft.ifftshift(freq), dim=(tuple(range(2, imgs.ndim))), norm='ortho')

        freq_loss = self.perceptual_loss(1 - preds, freq).mean()

        loss = self.lpips_weight * lpips_loss + self.bdcn_weight * bdcn_loss + self.freq_weight * freq_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/lpips_loss".format(split): lpips_loss.detach().mean(),
               "{}/bdcn_loss".format(split): bdcn_loss.detach().mean(),
               "{}/freq_loss".format(split): freq_loss.detach().mean(),
               }

        return loss, log






