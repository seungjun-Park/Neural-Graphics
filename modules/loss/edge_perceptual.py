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
                 perceptual_weight: float = 1.0,
                 edge_image_perceptual_weight = 1.0,
                 ):

        super().__init__()
        self.pn_weight = pn_weight
        self.perceptual_weight = perceptual_weight
        self.edge_image_perceptual_weight = edge_image_perceptual_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train") -> torch.Tensor:
        pn = pn_loss(targets, inputs)

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean()
        boundary_loss = p_loss * self.perceptual_weight + pn * self.pn_weight
        boundary_weight = torch.clamp(2.0 / (self.eips(conds.contiguous(), inputs.contiguous()) + 1e-5).mean(), min=1e-4)

        eip_loss = torch.clamp(self.eips(conds.contiguous(), targets.contiguous()).mean(), min=2.0)

        loss = eip_loss * self.edge_image_perceptual_weight + boundary_loss * boundary_weight

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/pn_loss".format(split): pn.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/eip_loss".format(split): eip_loss.detach(),
               }

        return loss, log
