import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator
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

        self.lpips = LPIPS().eval()
        self.eips = EIPS(**eips_config).eval()
        for p in self.eips.parameters():
            p.requires_grad = False

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, training: bool = False, opt_idx: int = 0,
                global_step: int = 0) -> Tuple:
        split = 'train' if training else 'val'

        cats = cats_loss(prediction=preds, label=labels, weights=self.cats_weight).mean()

        l1_loss = F.l1_loss(preds, labels, reduction='mean')

        p_loss = self.lpips(preds.repeat(1, 3, 1, 1).contiguous(), labels.repeat(1, 3, 1, 1).contiguous()).mean()

        eips_loss = self.eips(img=imgs, edge=preds.repeat(1, 3, 1, 1).contiguous()).mean()

        loss = p_loss * self.lpips_weight + l1_loss * self.l1_weight + self.eips_weight + eips_loss + cats

        log = {"{}/loss".format(split): loss.clone().detach(),
               "{}/l1_loss".format(split): l1_loss.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/eips_loss".format(split): eips_loss.detach().mean(),
               "{}/cats_loss".format(split): cats.detach().mean(),
               }

        return loss, log




