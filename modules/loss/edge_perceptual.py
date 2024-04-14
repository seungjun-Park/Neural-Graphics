import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight, LPIPS
from models.gan.discriminator import Discriminator
from utils import cats_loss, bdcn_loss2, contents_loss


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 contents_weight=1.0,
                 cats_weight=(1., 0, 0),
                 l1_weight=1.0,
                 perceptual_weight=1.0, ):

        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.cats_weight = tuple(cats_weight)
        self.contents_weight = contents_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, split: str = "train") -> torch.Tensor:
        rec_loss = torch.abs(inputs.contiguous() - targets.contiguous())
        loss = rec_loss

        cats = cats_loss(targets.contiguous(), inputs.contiguous(), self.cats_weight)
        loss = loss + cats

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)

        p_loss = self.perceptual_loss(inputs.contiguous(), targets.contiguous())
        loss = loss + self.perceptual_weight * p_loss

        c_loss = self.perceptual_loss(conds.contiguous(), targets.contiguous())
        loss = loss + self.contents_weight * c_loss

        loss = torch.sum(loss) / loss.shape[0]
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        cats = torch.sum(cats) / cats.shape[0]
        c_loss = torch.sum(c_loss) / c_loss.shape[0]
        p_loss = torch.sum(p_loss) / p_loss.shape[0]

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/rec_loss".format(split): rec_loss.detach(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/contents_loss".format(split): c_loss.detach().mean(),
               "{}/cats_loss".format(split): cats.detach().mean(),
               }

        return loss, log