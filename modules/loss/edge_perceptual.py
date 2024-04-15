import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight, LPIPS
from models.gan.discriminator import Discriminator
from utils import cats_loss, bdcn_loss2


class EdgePerceptualLoss(nn.Module):
    def __init__(self,
                 cats_weight=(1., 0, 0),
                 edge_weight=1.0,
                 contents_weight=1.0,
                 ):

        super().__init__()
        self.edge_weight = edge_weight
        self.perceptual_loss = LPIPS().eval()
        self.cats_weight = tuple(cats_weight)
        self.contents_weight = contents_weight

    def calculate_adaptive_weight(self, edge_loss, content_loss, last_layer):
        edge_grads = torch.autograd.grad(edge_loss, last_layer, retain_graph=True)[0]
        contents_grads = torch.autograd.grad(content_loss, last_layer, retain_graph=True)[0]

        contents_weight = torch.norm(edge_grads) / (torch.norm(contents_grads) + 1e-4)
        contents_weight = torch.clamp(contents_weight, 0.0, self.contents_weight).detach()
        return contents_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, conds: torch.Tensor, last_layer, split: str = "train") -> torch.Tensor:
        cats = cats_loss(targets.contiguous(), inputs.contiguous(), self.cats_weight)
        cats = torch.sum(cats) / cats.shape[0]

        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        if targets.shape[1] == 1:
            targets = targets.repeat(1, 3, 1, 1)
        if conds.shape[1] == 3:
            conds = 0.299 * conds[:, 0, :, :] + 0.587 * conds[:, 1, :, :] + 0.114 + conds[:, 2, :, :] # RGB to Grayscale
            conds = conds.unsqueeze(1).repeat(1, 3, 1, 1)

        l1_loss = torch.abs(inputs.contiguous() - targets.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), targets.contiguous())
        edge_loss = l1_loss + p_loss
        edge_loss = torch.sum(edge_loss) / edge_loss.shape[0]

        contents_p_loss = self.perceptual_loss(conds.contiguous(), targets.contiguous())
        contents_l1_loss = torch.abs(conds.contiguous() - targets.contiguous())
        contents_loss = contents_l1_loss + contents_p_loss
        contents_loss = torch.sum(contents_loss) / contents_loss.shape[0]

        if self.training and self.contents_weight > 0:
            contents_weight = self.calculate_adaptive_weight(edge_loss, contents_loss, last_layer)
        else:
            contents_weight = torch.tensor(0.0)

        loss = edge_loss * self.edge_weight + contents_loss * contents_weight + cats

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/edge_loss".format(split): edge_loss.detach(),
               "{}/contents_loss".format(split): contents_loss.detach(),
               "{}/cats_loss".format(split): cats.detach(),
               "{}/contents_weight".format(split): contents_weight.detach()
               }

        return loss, log