import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss, wasserstein_d_loss, LFD
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_config: DictConfig,
                 disc_start: int, disc_factor: float = 1.0, disc_weight:float = 1.0,
                 perceptual_weight: float = 1.0, l1_weight: float = 1.0, edge_perceptual_weight: float = 1.0,
                 lfd_weight: float = 1.0,):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.edge_perceptual_weight = edge_perceptual_weight
        self.lfd_weight = lfd_weight

        # self.discriminator = Discriminator(**disc_config)
        # self.discriminator_iter_start = disc_start
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, rec_loss: torch.Tensor, g_loss: torch.Tensor, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-5)
        d_weight = torch.clamp(d_weight, 1e-5, 1e5).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor,
                optimizer_idx: int = 0,
                global_step: int = 0,
                last_layer=None,
                split="train"):
        # now the GAN part
        if optimizer_idx == 0:
            l1_loss = F.l1_loss(preds, labels).mean()

            preds = preds.repeat(1, 3, 1, 1).contiguous()
            labels = labels.repeat(1, 3, 1, 1).contiguous()

            p_loss = self.perceptual_loss(preds, labels).mean()
            loss = self.l1_weight * l1_loss + self.perceptual_weight * p_loss

            # generator update
            # logits_fake = self.discriminator(imgs=imgs, edges=preds, training=False)['logits']
            # g_loss = -torch.mean(logits_fake)

            # if self.disc_factor > 0.0:
            #     try:
            #         d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
            #     except RuntimeError:
            #         assert not self.training
            #         d_weight = torch.tensor(0.0)
            # else:
            #     d_weight = torch.tensor(0.0)
            #
            # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # loss = rec_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/l1_loss".format(split): l1_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   # "{}/d_weight".format(split): d_weight.detach(),
                   # "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   # "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            imgs = imgs.detach().contiguous()
            labels = labels.detach().repeat(1, 3, 1, 1).contiguous()
            preds = preds.detach().repeat(1, 3, 1, 1).contiguous()

            real = self.discriminator(imgs=imgs, edges=labels, training=True)
            fake = self.discriminator(imgs=imgs, edges=preds, training=True)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            hinge = hinge_d_loss(real['logits'], fake['logits'])
            wasserstein = wasserstein_d_loss(real['dir'], fake['dir'])

            d_loss = disc_factor * (hinge + wasserstein)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): real['logits'].detach().mean(),
                   "{}/logits_fake".format(split): fake['logits'].detach().mean(),
                   "{}/dir_real".format(split): real['dir'].detach().mean(),
                   "{}/dir_fake".format(split): fake['dir'].detach().mean(),
                   }
            return d_loss, log






