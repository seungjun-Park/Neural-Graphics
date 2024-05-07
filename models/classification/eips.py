import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img


class EIPS(pl.LightningModule):
    def __init__(self,
                 net_config: dict,
                 criterion_config: dict,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 use_fp16: bool = False,
                 use_deep_supervision: bool = False,
                 mean_img: Union[List[float], Tuple[float]] = (0.5965, 0.5498, 0.5482),
                 std_img: Union[List[float], Tuple[float]] = (0.2738, 0.2722, 0.2641),
                 mean_edge: float = 0.9085,
                 std_edge: float = 0.2184,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.use_deep_supervision = use_deep_supervision
        self.use_fp16 = use_fp16

        self.register_buffer('mean_img', torch.Tensor(mean_img))
        self.register_buffer('std_img', torch.Tensor(std_img))
        self.register_buffer('mean_edge', torch.Tensor(to_3tuple(mean_edge)))
        self.register_buffer('std_edge', torch.Tensor(to_3tuple(std_edge)))

        self.net = instantiate_from_config(net_config)
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

    def normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        img = (img - self.mean_img[None, :, None, None]) / self.std_img[None, :, None, None]

        return img

    def normalize_edge(self, edge: torch.Tensor) -> torch.Tensor:
        edge = (edge - self.mean_edge[None, :, None, None]) / self.std_edge[None, :, None, None]

        return edge

    def forward(self, img: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img = self.normalize_img(img)
            edge = self.normalize_edge(edge)
        feat_img = self.net(img, self.use_deep_supervision)
        feat_edge = self.net(edge, self.use_deep_supervision)

        return self.criterion(feat_img, feat_edge)

    def training_step(self, batch, batch_idx):
        img, edge, tag = batch

        loss = self(img, edge)

        self.log('train/loss', loss, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, tag = batch

        loss = self(img, edge)

        self.log('val/loss', loss, logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        params = list(self.net.parameters())
        if isinstance(self.criterion, nn.Module):
            params = params + list(self.criterion.parameters())

        opt = torch.optim.AdamW(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        return [opt]