import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight
from modules.loss.lpips import LPIPS
from utils import FD


class LPIPSWithFD(nn.Module):
    def __init__(self, lpips_config, kl_weight=1.0, perceptual_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS(**lpips_config).eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, outputs, split="train"):
        rec_loss = torch.abs(outputs['freq'].contiguous() - outputs['recon_freq'].contiguous())
        # rec_loss = FD(inputs.contiguous(), reconstructions.contiguous())

        # p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        # rec_loss = rec_loss + self.perceptual_weight * p_loss
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # kl_loss = outputs['posterior'].kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = rec_loss # + self.kl_weight * kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               # "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               # "{}/p_loss".format(split): p_loss.detach().mean(),
               }

        return loss, log
