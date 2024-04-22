import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img
from utils.loss import normalized_euclidean_distance, cosine_distance
from modules.blocks import (
    ResidualBlock, DownBlock,
    WindowAttnBlock, MLP
)


class EIPS(pl.LightningModule):
    def __init__(self,
                 net_config: dict,
                 criterion_config: dict,
                 margin: float = 1.0,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval

        self.net = instantiate_from_config(net_config)
        self.criterion = instantiate_from_config(criterion_config)

        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=self.criterion,
                                                     margin=margin)

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

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
        feat0 = self.net(in0)
        feat1 = self.net(in1)

        return self.criterion(feat0, feat1)

    def training_step(self, batch, batch_idx):
        img, edge_pos, edge_neg = batch

        feat_anc = self.net(img)
        feat_pos = self.net(edge_pos)
        feat_neg = self.net(edge_neg)

        loss = self.loss(feat_anc, feat_pos, feat_neg)
        dist_pos = self.criterion(feat_anc, feat_pos).detach().mean()
        dist_neg = self.criterion(feat_anc, feat_neg).detach().mean()

        self.log('train/loss', loss, logger=True, rank_zero_only=True)
        self.log('train/dist_pos', dist_pos, logger=True, rank_zero_only=True)
        self.log('train/dist_neg', dist_neg, logger=True, rank_zero_only=True)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, edge_pos, edge_neg)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, edge_pos, edge_neg = batch

        feat_anc = self(img)
        feat_pos = self(edge_pos)
        feat_neg = self(edge_neg)

        dist_pos = self.criterion(feat_anc, feat_pos)
        dist_neg = self.criterion(feat_anc, feat_neg)

        loss = self.loss(feat_anc, feat_pos, feat_neg)

        self.log('val/loss', loss, logger=True, rank_zero_only=True)
        self.log('val/dist_pos', dist_pos, logger=True, rank_zero_only=True)
        self.log('val/dist_neg', dist_neg, logger=True, rank_zero_only=True)

        self.log_img(img, edge_pos, edge_neg)

    @torch.no_grad()
    def log_img(self, img, edge_pos, edge_neg):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', img[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge_pos', edge_pos[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge_neg', edge_neg[0], self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.encoder.parameters()) +
                                list(self.dense_layers.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        # lr_net = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=opt_net,
        #     lr_lambda=lambda epoch: 1.0 if epoch < self.lr_decay_epoch else (0.95 ** (epoch - self.lr_decay_epoch))
        # )

        return [opt]#, [{"scheduler": lr_net, "interval": "epoch"}]