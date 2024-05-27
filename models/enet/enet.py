import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img, to_rgb


class EDNSE(pl.LightningModule):
    def __init__(self,
                 net_config: dict,
                 loss_config: dict = None,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 use_fp16: bool = False,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.log_val = True
        # self.save_hyperparameters()

        self._dtype = torch.float16 if use_fp16 else torch.float32

        self.net = instantiate_from_config(net_config)

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return F.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx) -> Optional[Any]:
        img, label, cond = batch
        pred = self(img)

        loss, loss_log = self.loss(pred, label, img, split='train')

        if self.global_step % self.log_interval == 0:
            self.log_img(img, label, pred)

        self.log_dict(loss_log, rank_zero_only=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, label, cond = batch
        pred = self(img)

        loss, loss_log = self.loss(pred, label, img, split='val')

        if self.log_val:
            self.log_val = False
            self.log_img(img, label, pred)

        self.log_dict(loss_log, rank_zero_only=True, logger=True)

    def on_validation_end(self):
        self.log_val = True

    @torch.no_grad()
    def log_img(self, img, gt, pred):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', img[0], self.global_step // 2, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', gt[0], self.global_step // 2, dataformats='CHW')
        tb.add_image(f'{prefix}/pred', pred[0], self.global_step // 2, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.net.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        return [opt_net]
