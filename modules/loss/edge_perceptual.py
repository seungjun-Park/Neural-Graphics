import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2, existence_loss, strength_loss, smoothing_loss
from taming.modules.losses import LPIPS
from models.classification import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 existence_weight: float = 1.1,
                 strength_weight: float = 1.0,
                 smoothing_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 ):

        super().__init__()
        self.existence_weight = existence_weight
        self.lpips_weight = lpips_weight
        self.strength_weight = strength_weight
        self.smoothing_weight = smoothing_weight

        self.lpips = LPIPS().eval()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, split: str = "train", threshold: float = 0.8) -> torch.Tensor:
        existence = existence_loss(preds, labels, threshold=threshold) * self.existence_weight
        strength = strength_loss(preds, labels) * self.strength_weight
        smoothing = smoothing_loss(preds, labels) * self.smoothing_weight

        if preds.shape[1] == 1:
            preds = preds.repeat(1, 3, 1, 1)

        if labels.shape[1] == 1:
            labels = labels.repeat(1, 3, 1, 1)

        p_loss = self.lpips(preds.contiguous(), labels.contiguous()).mean() * self.lpips_weight

        loss = p_loss + existence + strength + smoothing

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/existence_loss".format(split): existence.detach().mean(),
               "{}/strength_loss".format(split): strength.detach().mean(),
               "{}/smoothing_loss".format(split): smoothing.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               }

        return loss, log
