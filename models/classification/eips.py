import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from models.classification.transformer import SwinTransformer


class EIPS(pl.LightningModule):
    def __init__(self,
                 net_config: dict,
                 margin: float = 1.0,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 dim: int = 2,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.margin = margin

        self.net: SwinTransformer = instantiate_from_config(net_config).eval()

        self.criterion = nn.ModuleList()

        hidden_dims = list(net_config['params']['hidden_dims'])

        for hidden_dim in hidden_dims:
            self.criterion.append(
                conv_nd(dim, hidden_dim, 1, kernel_size=1, stride=1)
            )

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

    def forward(self, img: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        img_feats = self.net.feature_extract(img, True)
        edge_feats = self.net.feature_extract(edge, True)

        dists = []

        for i, module in enumerate(self.criterion):
            dists.append(torch.mean(module((img_feats[i] - edge_feats[i]) ** 2), dim=[-2, -1], keepdim=True))

        dist = dists[0]

        for i in range(1, len(dists)):
            dist += dists[i]

        return dist.mean()

    def training_step(self, batch, batch_idx):
        img, edge_pos, edge_neg = batch

        anc_feats = self.net.feature_extract(img, True)
        pos_feats = self.net.feature_extract(edge_pos, True)
        neg_feats = self.net.feature_extract(edge_neg, True)

        pos_dists = []
        neg_dists = []

        for i, module in enumerate(self.criterion):
            pos_dists.append(torch.mean(module((anc_feats[i] - pos_feats[i]) ** 2), dim=[-2, -1], keepdim=True))
            neg_dists.append(torch.mean(module((anc_feats[i] - neg_feats[i]) ** 2), dim=[-2, -1], keepdim=True))

        pos_dist = pos_dists[0]
        neg_dist = neg_dists[0]

        for i in range(1, len(pos_dists)):
            pos_dist += pos_dists[i]
            neg_dist += neg_dists[i]

        loss = torch.clamp_min(self.margin + pos_dist - neg_dist, 0).mean()

        self.log('train/loss', loss, logger=True, rank_zero_only=True)
        self.log('train/pos_dist', pos_dist.mean(), logger=True, rank_zero_only=True)
        self.log('train/neg_dist', neg_dist.mean(), logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge_pos, edge_neg = batch

        anc_feats = self.net.feature_extract(img, True)
        pos_feats = self.net.feature_extract(edge_pos, True)
        neg_feats = self.net.feature_extract(edge_neg, True)

        pos_dists = []
        neg_dists = []

        for i, module in enumerate(self.criterion):
            pos_dists.append(torch.mean(module((anc_feats[i] - pos_feats[i]) ** 2), dim=[-2, -1], keepdim=True))
            neg_dists.append(torch.mean(module((anc_feats[i] - neg_feats[i]) ** 2), dim=[-2, -1], keepdim=True))

        pos_dist = pos_dists[0]
        neg_dist = neg_dists[0]

        for i in range(1, len(pos_dists)):
            pos_dist += pos_dists[i]
            neg_dist += neg_dists[i]

        loss = torch.clamp_min(self.margin + pos_dist - neg_dist, 0).mean()

        self.log('val/loss', loss, logger=True, rank_zero_only=True)
        self.log('val/pos_dist', pos_dist.mean(), logger=True, rank_zero_only=True)
        self.log('val/neg_dist', neg_dist.mean(), logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        params = list(self.criterion.parameters())

        opt = torch.optim.AdamW(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        return [opt]