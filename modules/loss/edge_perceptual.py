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


def get_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, *spatial = x.shape
    x = x.reshape(b, c, -1)
    gram = torch.bmm(x, x.transpose(1, 2))
    return gram


class EdgeLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 content_weight: float = 1.0,
                 style_weight: float = 1.0,
                 recon_weight: float = 1.0,
                 ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.recon_weight = recon_weight

        self.learnable_weights = [0.75, 0.5, 0.25, 0.25, 0.25]

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, imgs: torch.Tensor, split="train"):
        preds = preds.repeat(1, 3, 1, 1).contiguous()
        labels = labels.repeat(1, 3, 1, 1).contiguous()

        content_loss = self.perceptual_loss(preds, imgs).mean()
        feats_labels, feats_preds = self.vgg16(labels), self.vgg16(preds)
        style_loss = torch.tensor(0.0).to(preds.device)

        for i in range(len(self.learnable_weights)):
            gram_labels = get_gram_matrix(feats_labels[i])
            gram_preds = get_gram_matrix(feats_preds[i])
            style_loss += self.learnable_weights[i] * F.mse_loss(gram_preds, gram_labels) / np.prod(feats_labels[i].shape)

        recon_loss = self.perceptual_loss(preds, labels).mean()

        loss = self.content_weight * content_loss + self.style_weight * style_loss + self.recon_weight * recon_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/content_loss".format(split): content_loss.detach().mean(),
               "{}/style_loss".format(split): style_loss.detach().mean(),
               "{}/recon_loss".format(split): recon_loss.detach().mean(),
               }

        return loss, log






