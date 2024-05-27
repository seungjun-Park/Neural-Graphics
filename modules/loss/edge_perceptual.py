import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, existence_loss, strength_loss, smoothing_loss, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss
from taming.modules.losses import LPIPS
from models.classification.eips import EIPS


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 eips_config: DictConfig,
                 bdcn_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 eips_weight: float = 1.0,
                 ):

        super().__init__()
        self.bdcn_weight = bdcn_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.eips_weight = eips_weight

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split: str = "train") -> torch.Tensor:
        bdcn = bdcn_loss2(preds, labels)
        l1 = F.l1_loss(preds, labels, reduction='mean')

        p_loss = self.lpips(preds.repeat(1, 3, 1, 1).contiguous(), labels.repeat(1, 3, 1, 1).contiguous()).mean()

        eips = self.eips(img=imgs.contiguous(), edge=preds.repeat(1, 3, 1, 1).contiguous()).mean()

        loss = p_loss * self.lpips_weight + l1 * self.l1_weight + bdcn * self.bdcn_weight + eips * self.eips_weight

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/l1_loss".format(split): l1.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/eips".format(split): eips.detach().mean(),
               "{}/bdcn_loss".format(split): bdcn.detach().mean(),
               }

        return loss, log



