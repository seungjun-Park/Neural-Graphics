import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2
from taming.modules.losses import LPIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 contents_weight: float = 0.0,
                 cats_weight: Union[List[float], Tuple[float]] = (1., 0., 0.),
                 bdcn_weight: float = 1.1,
                 ):

        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.contents_weight = contents_weight
        self.cats_weight = cats_weight
        self.bdcn_weight = bdcn_weight

        self.lpips = LPIPS().eval()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train") -> torch.Tensor:
        l1_loss = F.l1_loss(inputs.contiguous(), targets.contiguous(), reduction='mean')
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)

        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.lpips(inputs.contiguous(), targets.contiguous()).mean()

        edge_loss = l1_loss * self.l1_weight + p_loss * self.perceptual_weight

        contents_loss = self.lpips(conds.contiguous(), targets.contiguous()).mean()

        cats = cats_loss(targets, inputs, self.cats_weight).mean()
        bdcn = bdcn_loss2(targets, inputs, self.bdcn_weight).mean()

        loss = edge_loss + cats + bdcn + contents_loss * self.contents_weight

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/edge_loss".format(split): edge_loss.detach(),
               "{}/l1_loss".format(split): l1_loss.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/cats_loss".format(split): cats.detach(),
               "{}/bdcn_loss2".format(split): bdcn.detach(),
               "{}/contents_loss".format(split): contents_loss.detach(),
               }

        return loss, log
