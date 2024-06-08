import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.blocks.encoder import SwinEncoder
from modules.blocks.decoder import SwinDecoder


class EIPS(pl.LightningModule):
    def __init__(self,
                 net_config: DictConfig,
                 criterion_config: DictConfig,
                 margin: float = 1.0,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.margin = margin

        self.encoder = SwinEncoder(**net_config)
        self.logit = nn.Linear(int(self.encoder.quant_dim * (self.encoder.cur_res ** 2)), 1)
        self.criterion = instantiate_from_config(criterion_config)

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def train(self, mode: bool = True):
        super().train(mode)
        for param in self.parameters():
            param.requires_grad = True

        return self

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        feat0 = self.encoder(x0)
        feat1 = self.encoder(x1)

        return self.criterion(feat0, feat1)

    def training_step(self, batch, batch_idx):
        anc, pos, neg, label_pos, label_neg = batch

        feat_anc = self.encoder(anc)
        feat_pos = self.encoder(pos)
        feat_neg = self.encoder(neg)
        triplet_loss = F.triplet_margin_with_distance_loss(feat_anc, feat_pos, feat_neg, distance_function=self.criterion, margin=self.margin)
        logit_pos = self.logit(torch.flatten(feat_pos, start_dim=1))
        logit_neg = self.logit(torch.flatten(feat_neg, start_dim=1))
        logit = torch.cat([logit_pos, logit_neg], dim=0)
        bce_loss = F.binary_cross_entropy_with_logits(logit, torch.cat([label_pos, label_neg], dim=0))

        loss = bce_loss + triplet_loss

        dist_pos = self.criterion(anc, pos).mean()
        dist_neg = self.criterion(anc, neg).mean()

        self.log('train/loss', loss, logger=True, rank_zero_only=True)
        self.log('train/bce_loss', bce_loss, logger=True, rank_zero_only=True)
        self.log('train/triplet_loss', triplet_loss, logger=True, rank_zero_only=True)
        self.log('train/dist_pos', dist_pos, logger=True, rank_zero_only=True)
        self.log('train/dist_neg', dist_neg, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        anc, pos, neg, label_pos, label_neg = batch

        feat_anc = self.encoder(anc)
        feat_pos = self.encoder(pos)
        feat_neg = self.encoder(neg)
        triplet_loss = F.triplet_margin_with_distance_loss(feat_anc, feat_pos, feat_neg,
                                                           distance_function=self.criterion, margin=self.margin)
        logit_pos = self.logit(feat_pos)
        logit_neg = self.logit(feat_neg)
        logit = torch.cat([logit_pos, logit_neg], dim=0)
        bce_loss = F.binary_cross_entropy_with_logits(logit, torch.cat([label_pos, label_neg], dim=0))

        loss = bce_loss + triplet_loss

        dist_pos = self.criterion(anc, pos).mean()
        dist_neg = self.criterion(anc, neg).mean()

        self.log('val/loss', loss, logger=True, rank_zero_only=True)
        self.log('val/bce_loss', bce_loss, logger=True, rank_zero_only=True)
        self.log('val/triplet_loss', triplet_loss, logger=True, rank_zero_only=True)
        self.log('val/dist_pos', dist_pos, logger=True, rank_zero_only=True)
        self.log('val/dist_neg', dist_neg, logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        params = list(self.encoder.parameters()) + list(self.logit.parameters())
        if isinstance(self.criterion, nn.Module):
            params += list(self.criterion.parameters())

        opt = torch.optim.AdamW(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        return [opt]