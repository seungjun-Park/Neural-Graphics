import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2, bdcn_loss3
from taming.modules.losses import LPIPS
from models.classification import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 eips_config,
                 bdcn_weight: float = 1.1,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 edge_image_perceptual_weight: float = 1.0,
                 ):

        super().__init__()
        self.bdcn_weight = bdcn_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.edge_image_perceptual_weight = edge_image_perceptual_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train",
                threshold: float = 0.5) -> torch.Tensor:
        bdcn = bdcn_loss3(targets, inputs, threshold=threshold) * self.bdcn_weight
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean() * self.perceptual_weight
        eips_weight = 1. / (self.eips(conds.contiguous(), inputs.contiguous()) + 1e-5).mean()

        loss = (p_loss + bdcn) * eips_weight

        eips = self.eips(conds.detach(), targets.detach())

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/bdcn_loss".format(split): bdcn.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/eips".format(split): eips.detach(),
               }

        return loss, log
