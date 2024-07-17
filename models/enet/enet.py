import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img, to_rgb
from modules.loss.edge_perceptual import EdgeLPIPSWithDiscriminator


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
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval

        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(self.net(x))

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        opt_net, opt_disc = self.optimizers()

        d_loss, d_loss_log = self.loss(preds, labels, imgs, 1, self.global_step, last_layer=self.get_last_layer(), split='train')
        d_loss = d_loss / self.accumulate_grad_batches
        self.manual_backward(d_loss)
        net_loss, net_loss_log = self.loss(preds, labels, imgs, 0, self.global_step, last_layer=self.get_last_layer(), split='train')
        net_loss = net_loss / self.accumulate_grad_batches
        self.manual_backward(net_loss)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_disc.step()
            opt_disc.zert_grad()
            opt_net.step()
            opt_net.zero_grad()

        if self.global_step % self.log_interval == 0:
            self.log_img(preds, 'edge')
            self.log_img(labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(d_loss_log)
        self.log_dict(net_loss_log)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        d_loss, d_loss_log = self.loss(preds, labels, imgs, 1, self.global_step, last_layer=self.get_last_layer(), split='val')
        net_loss, net_loss_log = self.loss(preds, labels, imgs, 0, self.global_step, last_layer=self.get_last_layer(), split='val')

        if self.global_step % self.log_interval == 0:
            self.log_img(preds, 'edge')
            self.log_img(labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(d_loss_log)
        self.log_dict(net_loss_log)

    def get_last_layer(self):
        return self.net.out[-1].weight

    @torch.no_grad()
    def log_img(self, x: torch.Tensor, split='img'):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/{split}', x[0], self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.net.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        opts = [opt_net]

        if isinstance(self.loss, EdgeLPIPSWithDiscriminator):
            opt_disc = torch.optim.AdamW(list(self.loss.discriminator.parameters()),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay,
                                         betas=(0.5, 0.9),
                                         )
            opts.append(opt_disc)

        return opts
