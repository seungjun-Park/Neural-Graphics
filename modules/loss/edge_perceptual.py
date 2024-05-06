import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2, pn_loss
from taming.modules.losses import LPIPS
from models.classification import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 eips_config,
                 pn_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 edge_image_perceptual_weight: float = 1.0,
                 ):

        super().__init__()
        self.pn_weight = pn_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.edge_image_perceptual_weight = edge_image_perceptual_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train") -> torch.Tensor:
        l1 = F.l1_loss(inputs, targets) * self.l1_weight
        pn = pn_loss(targets, inputs) * self.pn_weight
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean() * self.perceptual_weight
        eips_weight = torch.clamp(1. / (self.eips(conds.contiguous(), inputs.contiguous()) + 1e-5).mean(), min=1e-5, max=1e2)

        loss = (p_loss + pn + l1) * eips_weight

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/pn_loss".format(split): pn.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/l1_loss".format(split): l1.detach(),
               }

        return loss, log
