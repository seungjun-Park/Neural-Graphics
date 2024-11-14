import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from omegaconf import DictConfig
from utils import cats_loss, bdcn_loss2, adopt_weight
from utils.loss import hinge_d_loss, vanilla_d_loss, san_d_loss, wasserstein_d_loss, LFD
from taming.modules.losses import LPIPS
from models.gan.discriminator import Discriminator


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 cats_weight: List[float] = (1.0, 0.0, 0.0)
                 ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.cats_weight = cats_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split="train"):
        # preds = preds.repeat(1, 3, 1, 1).contiguous()
        # labels = labels.repeat(1, 3, 1, 1).contiguous()
        #
        # content_loss = self.perceptual_loss(preds, imgs).mean()
        # feats_labels, feats_preds = self.vgg16(labels), self.vgg16(preds)
        # style_loss = torch.tensor(0.0).to(preds.device)

        preds = 1 - preds
        labels = 1 - labels

        loss, tracing_loss, bdr_loss, texture_loss = cats_loss(preds, labels, self.cats_weight)

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/tracing_loss".format(split): tracing_loss.detach().mean(),
               "{}/bdr_loss".format(split): bdr_loss.detach().mean(),
               "{}/texture_loss".format(split): texture_loss.detach().mean(),
               }

        return loss, log






