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
                 bdcn_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 edge_image_perceptual_weight = 1.0,
                 ):

        super().__init__()
        self.bdcn_weight = bdcn_weight
        self.perceptual_weight = perceptual_weight
        self.edge_image_perceptual_weight = edge_image_perceptual_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train") -> torch.Tensor:
        bdcn_loss = bdcn_loss3(inputs.contiguous(), targets.contiguous()).mean()

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean()
        eip_loss = self.eips(conds.contiguous(), targets.contiguous()).mean()

        loss = p_loss * self.perceptual_weight + eip_loss * self.edge_image_perceptual_weight + bdcn_loss * self.bdcn_weight

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/bdcn_loss".format(split): bdcn_loss.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/eip_loss".format(split): eip_loss.detach(),
               }

        return loss, log
