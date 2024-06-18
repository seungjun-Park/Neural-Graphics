import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss
from taming.modules.losses import LPIPS
from models.gan.discriminator import SwinDiscriminator


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 disc_config: DictConfig,
                 cats_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 disc_weight: float = 1.0,
                 disc_start_iter: int = 0,
                 ):

        super().__init__()
        self.cats_weight = cats_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.disc_weight = disc_weight
        self.disc_start_iter = disc_start_iter

        self.lpips = LPIPS().eval()
        self.disc = SwinDiscriminator(**disc_config)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, training: bool = False, opt_idx: int = 0,
                global_step: int = 0) -> torch.Tensor:
        split = 'train' if training else 'val'

        if opt_idx == 0:
            l1_loss = F.l1_loss(preds, labels, reduction='mean')

            p_loss = self.lpips(preds.repeat(1, 3, 1, 1).contiguous(), labels.repeat(1, 3, 1, 1).contiguous()).mean()

            g_loss = -torch.mean(self.disc(imgs.contiguous(), preds.repeat(1, 3, 1, 1)))

            g_weight = adopt_weight(self.disc_weight, global_step, self.disc_start_iter)

            loss = p_loss * self.lpips_weight + l1_loss * self.l1_weight + g_weight * g_loss

            log = {"{}/loss".format(split): loss.clone().detach(),
                   "{}/l1_loss".format(split): l1_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   # "{}/cats_loss".format(split): cats.detach().mean(),
                   }

            return loss, log

        if opt_idx == 1:
            # second pass for discriminator update
            logits_real = self.disc(imgs=imgs, edges=labels.repeat(1, 3, 1, 1))
            logits_fake = self.disc(imgs=imgs, edges=preds.repeat(1, 3, 1, 1).detach())

            d_loss = hinge_d_loss(logits_real, logits_fake) * adopt_weight(1.0, global_step, self.disc_start_iter)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            return d_loss, log




