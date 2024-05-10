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
                 accumulate_grad_batches: int = 1,
                 disc_update_frequency: float = 1.0,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.disc_update_frequency = disc_update_frequency
        self.disc_accumulate_grad_batches = (self.accumulate_grad_batches // self.disc_update_frequency)

        self.log_val = True

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return F.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx) -> Optional[Any]:
        img, label, cond = batch

        opt_net, opt_disc = self.optimizers()

        pred = self(img)

        with torch.autograd.detect_anomaly():
            net_loss, net_loss_log = self.loss(pred, label, img, split='train', optimizer_idx=0,
                                               global_step=self.global_step)
            disc_loss, disc_loss_log = self.loss(pred, label, img, split='train', optimizer_idx=1,
                                                 global_step=self.global_step)
            net_loss = net_loss / self.accumulate_grad_batches
            disc_loss = disc_loss / self.disc_accumulate_grad_batches

            if (self.global_step // 2) % self.log_interval == 0:
                self.log_img(img, label, pred)

            self.log_dict(net_loss_log)
            self.log_dict(disc_loss_log)

            self.manual_backward(net_loss)

            if (batch_idx + 1) % self.accumulate_grad_batches:
                opt_net.step()
                opt_net.zero_grad()

            self.manual_backward(disc_loss)

            if (batch_idx + 1) % self.disc_accumulate_grad_batches:
                opt_disc.step()
                opt_disc.zero_grad()

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, label, cond = batch

        pred = self(img)

        net_loss, net_loss_log = self.loss(pred, label, img, split='val', optimizer_idx=0, global_step=self.global_step)
        disc_loss, disc_loss_log = self.loss(pred, label, img, split='val', optimizer_idx=1, global_step=self.global_step)

        if self.log_val:
            self.log_img(img, label, pred)
            self.log_val = False

        self.log_dict(net_loss_log)
        self.log_dict(disc_loss_log)

    def on_validation_end(self):
        self.log_val = True

    @torch.no_grad()
    def log_img(self, img, gt, pred):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', img[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', gt[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/pred', pred[0], self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.net.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        opt_disc = torch.optim.AdamW(list(self.loss.disc.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.9)
                                     )

        return [opt_net, opt_disc]
