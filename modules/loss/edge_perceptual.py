import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss, wasserstein_d_loss
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_config: DictConfig,
                 disc_start, logvar_init=0.0, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = Discriminator(**disc_config)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor,
                optimizer_idx: int = 0,
                global_step: int = 0,
                last_layer=None,
                split="train",
                weights=None):
        rec_loss = torch.abs(preds.contiguous() - labels.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(preds.contiguous(), labels.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(imgs.contiguous(), preds.contiguous())['logits']
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            real = self.discriminator(imgs.contiguous().detach(), labels.contiguous().detach())
            fake = self.discriminator(imgs.contiguous().detach(), preds.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            hinge = hinge_d_loss(real['logits'], fake['logits'])
            wasserstein = wasserstein_d_loss(real['dir'], fake['dir'])

            d_loss = disc_factor * (hinge + wasserstein)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): real['logits'].detach().mean(),
                   "{}/logits_fake".format(split): fake['logits'].detach().mean(),
                   "{}/dir_real".format(split): real['dir'].detach().mean(),
                   "{}/dir_fake".format(split): fake['dir'].detach().mean()
                   }
            return d_loss, log






