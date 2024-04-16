import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight, LPIPS
from models.gan.discriminator import Discriminator
from utils import cats_loss, bdcn_loss2


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 disc_config,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 disc_iter_start: int = 0,
                 d_weight: float = 1.0,
                 g_weight: float = 1.0,
                 disc_loss: str = 'hinge'
                 ):

        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = LPIPS().eval()
        self.disc = Discriminator(**disc_config)
        self.disc_iter_start = disc_iter_start
        self.d_weight = d_weight
        self.g_weight = g_weight

        disc_loss = disc_loss.lower()
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, edge_loss, g_loss, last_layer):
        edge_grads = torch.autograd.grad(edge_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        g_weight = torch.norm(edge_grads) / (torch.norm(g_grads) + 1e-4)
        g_weight = torch.clamp(g_weight, 0.0, 1e4).detach()
        return g_weight * self.g_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, global_step: int,
                last_layer, split: str = "train") -> torch.Tensor:

        l1_loss = torch.abs(inputs.contiguous() - targets.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), targets.contiguous())

        edge_loss = l1_loss * self.l1_weight + p_loss * self.perceptual_weight
        edge_loss = torch.sum(edge_loss) / edge_loss.shape[0]

        logits_fake = self.disc(torch.cat([targets.contiguous(), conds], dim=1))
        g_loss = -torch.mean(logits_fake)

        if self.training and self.g_weight > 0:
            g_weight = self.calculate_adaptive_weight(edge_loss, g_loss, last_layer)
        else:
            g_weight = torch.tensor(0.0)

        g_weight = adopt_weight(g_weight, global_step, threshold=self.disc_iter_start)
        loss = edge_loss + g_loss * g_weight

        logits_real = self.disc(torch.cat([inputs.contiguous().detach(), conds], dim=1))
        logits_fake = self.disc(torch.cat([targets.contiguous().detach(), conds], dim=1))

        d_loss = self.d_weight * self.disc_loss(logits_real, logits_fake)

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/edge_loss".format(split): edge_loss.detach(),
               "{}/g_loss".format(split): g_loss.detach(),
               "{}/logits_real".format(split): logits_real.detach().mean(),
               "{}/logits_fake".format(split): logits_fake.detach().mean(),
               "{}/d_loss".format(split): d_loss.detach().mean(),
               }

        return loss, d_loss, log
