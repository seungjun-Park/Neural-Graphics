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
                 l1_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 ):

        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

        self.lpips = LPIPS().eval()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, training: bool = False, split='') -> Tuple:
        prefix = 'train' if training else 'val'

        l1_loss = F.l1_loss(preds, labels, reduction='mean')

        p_loss = self.lpips(preds.repeat(1, 3, 1, 1).contiguous(), labels.repeat(1, 3, 1, 1).contiguous()).mean()

        loss = p_loss * self.lpips_weight + l1_loss * self.l1_weight

        log = {f"{prefix}/{split}/loss": loss.clone().detach(),
               f"{prefix}/{split}/l1_loss": l1_loss.detach().mean(),
               f"{prefix}/{split}/p_loss": p_loss.detach().mean(),
               }

        return loss, log






