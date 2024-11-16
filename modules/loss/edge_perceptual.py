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
                 cats_weight: List[float] = (1.0, 0.0, 1.2),
                 lpips_weight: float = 1.0,
                 balanced_mse_weight: float = 1.0,
                 bdcn_weight: float = 1.0,
                 ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.lpips_weight = lpips_weight
        self.balanced_mse_weight = balanced_mse_weight
        self.bdcn_weight = bdcn_weight
        self.cats_weight = cats_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split="train"):
        # cats, tracing_loss, bdr_loss, texture_loss = cats_loss(1 - preds, 1 - labels, self.cats_weight)

        # mask = labels.clone().detach()
        # mask = torch.round(mask).long().float()
        #
        # with torch.no_grad():
        #     balanced_w = 1.1
        #
        #     num_positive = torch.sum((mask == 1).float()).float()
        #     num_negative = torch.sum((mask == 0).float()).float()
        #     beta = num_negative / (num_positive + num_negative)
        #     mask[mask == 1] = beta
        #     mask[mask == 0] = balanced_w * (1 - beta)
        #     mask[mask == 2] = 0

        balanced_mse_loss = (F.mse_loss(preds, labels, reduction='none')).mean()

        preds = preds.repeat(1, 3, 1, 1).contiguous()
        labels = labels.repeat(1, 3, 1, 1).contiguous()

        lpips_loss = self.perceptual_loss(preds, labels).mean()

        # loss = self.lpips_weight * lpips_loss + cats
        loss = self.lpips_weight * lpips_loss + self.balanced_mse_weight * balanced_mse_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/lpips_loss".format(split): lpips_loss.detach().mean(),
               "{}/bmse_loss".format(split): balanced_mse_loss.detach().mean(),
               }

        return loss, log






