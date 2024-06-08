import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Union, List, Tuple, Dict

from utils import group_norm
from modules.blocks.encoder import SwinEncoder
from omegaconf import ListConfig, DictConfig


class SwinTransformer(pl.LightningModule):
    def __init__(self,
                 encoder_config: DictConfig,
                 lr: float = 1e-4,
                 num_class: int = 1000,
                 weight_decay: float = 0.,
                 ckpt_path: str = None,
                 ignored_keys: list = ['attn_mask']
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder = SwinEncoder(**encoder_config)
        self.num_class = num_class
        self.norm = group_norm(self.encoder.latent_dim, num_groups=encoder_config['num_groups'])
        self.logit_out = nn.Linear(self.encoder.latent_dim, num_class)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignored_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.norm(h)
        h = F.adaptive_avg_pool2d(h, 1)
        h = torch.flatten(h, start_dim=1)
        h = self.logit_out(h)

        return h

    @torch.no_grad()
    def feature_extract(self, x: torch.Tensor, is_deep_supervision: bool = False)\
            -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]:
        return self.encoder.feature_extract(x, is_deep_supervision)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        labels = F.one_hot(labels, num_classes=self.num_class).float()
        logit = self(x)
        prob = F.softmax(logit, dim=-1)
        loss = F.cross_entropy(prob, labels)

        self.log('train/loss', loss, rank_zero_only=True, logger=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        labels = F.one_hot(labels, num_classes=self.num_class).float()
        logit = self(x)
        prob = F.softmax(logit, dim=-1)
        loss = F.cross_entropy(prob, labels)

        self.log('val/loss', loss, rank_zero_only=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.99)
                                )

        return opt

