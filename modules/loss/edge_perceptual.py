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

    def forward(self, inputs, target, cond, split="train"):
        cats = cats_loss(target, inputs, self.cats_weight)
        rec_loss = F.l1_loss(inputs, target) * self.l1_weight
        loss = rec_loss
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        p_loss = self.perceptual_loss(inputs.contiguous(), target.contiguous())
        loss += self.perceptual_weight * torch.mean(p_loss)

        c_loss = self.perceptual_loss(cond.contiguous(), target.contiguous())
        loss += torch.mean(c_loss) * self.contents_weight

        loss += cats

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/p_loss".format(split): p_loss.detach().mean(),
               "{}/contents_loss".format(split): c_loss.detach().mean(),
               "{}/cats_loss".format(split): cats.detach().mean(),
               }

        return loss, log