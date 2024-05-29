import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.blocks.encoder import SwinEncoder
from modules.blocks.decoder import SwinDecoder


class EIPS(pl.LightningModule):
    def __init__(self,
                 net_config: dict,
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
        self.decoder = SwinDecoder(**net_config)

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
        cond = self.encoder(img)
        logit = self.decoder(edge, cond)

        return F.sigmoid(logit)

    def training_step(self, batch, batch_idx):
        img, edge, label = batch

        prob = self(img, edge)
        loss = F.binary_cross_entropy(prob, label)

        self.log('train/loss', loss, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, label = batch

        prob = self(img, edge)
        loss = F.binary_cross_entropy(prob, label)

        self.log('val/loss', loss, logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.encoder.parameters()) +
                                list(self.decoder.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.0, 0.99)
                                )

        return [opt]