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
                 lpips_weight: float = 1.0,
                 eips_weight: float = 1.0,
                 ):

        super().__init__()
        self.bdcn_weight = bdcn_weight
        self.lpips_weight = lpips_weight
        self.eips_weight = eips_weight
        self.l1_weight = l1_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train",
                threshold: float = 0.5) -> torch.Tensor:
        bdcn = bdcn_loss3(targets, inputs, threshold=threshold) * self.bdcn_weight
        l1 = F.l1_loss(inputs, targets, reduction='mean') * self.l1_weight
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean() * self.lpips_weight
        eips_loss = self.eips(conds.contiguous(), targets.contiguous()).mean()

        loss = p_loss + eips_loss * self.eips_weight + bdcn + l1

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/bdcn_loss".format(split): bdcn.detach(),
               "{}/l1_loss".format(split): l1.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/eips_loss".format(split): eips_loss.detach(),
               }

        return loss, log
