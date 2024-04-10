import torch
import torch.nn as nn

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
        rec_loss = torch.abs(inputs.contiguous() - target.contiguous()) * self.l1_weight

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), target.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        contents = self.perceptual_loss(target.contiguous(), cond.contiguous()) * self.contents_weight

        loss = cats + rec_loss + contents

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/contents_loss".format(split): contents.detach().mean(),
               "{}/cats_loss".format(split): cats.detach().mean(),
               }

        return loss, log