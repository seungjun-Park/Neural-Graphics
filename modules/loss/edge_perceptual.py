import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2, existence_loss, strength_loss, smoothing_loss, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 disc_config: dict,
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 d_loss_type: str = 'san',
                 g_weight: float = 1.0,
                 ):

        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.g_weight = g_weight
        self.d_loss_type = d_loss_type.lower()
        assert self.d_loss_type in ['vanilla', 'hinge', 'san']

        if self.d_loss_type == 'vanilla':
            self.d_loss = vanilla_d_loss
        elif self.d_loss_type == 'hinge':
            self.d_loss = hinge_d_loss
        elif self.d_loss_type == 'san':
            self.d_loss = san_d_loss

        self.lpips = LPIPS().eval()
        self.disc = Discriminator(**disc_config)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split: str = "train", optimizer_idx: int = 0) -> torch.Tensor:
        if optimizer_idx == 0:
            l1 = F.l1_loss(preds, labels, reduction='mean') * self.l1_weight

            p_loss = self.lpips(preds.contiguous().repeat(1, 3, 1, 1),
                                labels.contiguous().repeat(1, 3, 1, 1)).mean() * self.lpips_weight

            logits_fake = self.disc(torch.cat([preds, imgs], dim=1).contiguous(), training=False)
            g_loss = -logits_fake.mean() * self.g_weight

            loss = p_loss + l1 + g_loss

            log = {"{}/loss".format(split): loss.clone().detach(),
                   "{}/l1_loss".format(split): l1.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            out_real = self.disc(torch.cat([labels, imgs], dim=1).detach(), training=True)
            out_fake = self.disc(torch.cat([preds, imgs], dim=1).detach(), training=True)

            if self.d_loss_type == 'san':
                loss = self.d_loss(out_real=out_real, out_fake=out_fake)

                log = {"{}/disc_loss".format(split): loss.clone().detach().mean(),
                       "{}/logits_real".format(split): out_real['logits'].detach().mean(),
                       "{}/logits_fake".format(split): out_fake['logits'].detach().mean(),
                       "{}/dir_real".format(split): out_real['dir'].detach().mean(),
                       "{}/dir_fake".format(split): out_fake['dir'].detach().mean()
                       }

            else:
                loss = self.d_loss(out_real['logits'], out_fake['logits'])

                log = {"{}/disc_loss".format(split): loss.clone().detach().mean(),
                       "{}/logits_real".format(split): out_real['logits'].detach().mean(),
                       "{}/logits_fake".format(split): out_fake['logits'].detach().mean()
                       }

            return loss, log



