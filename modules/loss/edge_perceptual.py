import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss
from taming.modules.losses import LPIPS
from models.classification.eips import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 eips_config: DictConfig,
                 cats_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 eips_weight: float = 1.0,
                 ):

        super().__init__()
        self.cats_weight = cats_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.eips_weight = eips_weight

        # self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split: str = "train") -> torch.Tensor:
        # cats = cats_loss(preds, labels, self.cats_weight).mean()
        l2 = F.mse_loss(preds, labels, reduction='mean')

        p_loss = self.lpips(preds.repeat(1, 3, 1, 1).contiguous(), labels.repeat(1, 3, 1, 1).contiguous()).mean()

        eips = self.eips(imgs.contiguous(), preds.repeat(1, 3, 1, 1)).mean()

        loss = p_loss * self.lpips_weight + l2 * self.l1_weight + eips * self.eips_weight # + cats

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/l2_loss".format(split): l2.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/eips".format(split): eips.detach().mean(),
               # "{}/cats_loss".format(split): cats.detach().mean(),
               }

        return loss, log



