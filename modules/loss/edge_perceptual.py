import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import cats_loss, bdcn_loss2, existence_loss, strength_loss, smoothing_loss
from taming.modules.losses import LPIPS
from models.classification import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 cats_weight: Union[List[float], Tuple[float]] = (1.0, 0.0, 0.0),
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 ):

        super().__init__()
        self.l1_weight = l1_weight
        self.cats_weight = cats_weight
        self.lpips_weight = lpips_weight

        self.lpips = LPIPS().eval()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, split: str = "train", threshold: float = 0.8) -> torch.Tensor:
        l1 = F.l1_loss(preds, labels, reduction='mean') * self.l1_weight
        cats = cats_loss(preds, labels, weights=self.cats_weight).mean()

        if preds.shape[1] == 1:
            preds = preds.repeat(1, 3, 1, 1)

        if labels.shape[1] == 1:
            labels = labels.repeat(1, 3, 1, 1)

        p_loss = self.lpips(preds.contiguous(), labels.contiguous()).mean() * self.lpips_weight

        loss = p_loss + l1 + cats

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/cats_loss".format(split): cats.detach().mean(),
               "{}/l1_loss".format(split): l1.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               }

        return loss, log
