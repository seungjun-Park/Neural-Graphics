import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img, to_rgb
from modules.loss.edge_perceptual import EdgePerceptualLoss


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
        self.encoder = instantiate_from_config(net_config).encoder

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
        hs = []
        h = x
        for i, module in enumerate(self.encoder):
            h = module(h)
            hs.append(h)

        for i, block in enumerate(self.net.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        h = torch.cat([h, hs.pop()], dim=1)
        h = self.out(h)

        return h

    def training_step(self, batch, batch_idx):
        img, label, cond = batch

        opt_net, opt_encoder = self.optimizers()

        hs = []

        h = label.repeat(1, 3, 1, 1)
        for i, module in enumerate(self.net.encoder):
            h = module(h)
            hs.append(h)

        zs = []
        z = img
        for i, module in enumerate(self.encoder):
            z = module(z)
            zs.append(z)

        feat_losses = []
        for i, j in zip(hs, zs):
            feat_losses.append(F.mse_loss(i.detach(), j, reduction='none').mean(dim=[1, 2, 3]))
        feat_loss = sum(feat_losses).mean()

        for i, block in enumerate(self.net.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        with torch.no_grad():
            for i, block in enumerate(self.net.decoder):
                z = torch.cat([z, zs.pop()], dim=1)
                z = block(z)

        net_loss, net_loss_log = self.loss(h, label, training=True, split='net')
        net_loss /= self.accumulate_grad_batches
        self.manual_backward(net_loss)

        encoder_loss, encoder_loss_log = self.loss(z, label, training=True, split='encoder')
        encoder_loss = encoder_loss + feat_loss
        encoder_loss_log['train/encoder/loss'] = encoder_loss
        encoder_loss_log['train/encoder/feat_loss'] = feat_loss

        encoder_loss /= self.accumulate_grad_batches
        self.manual_backward(encoder_loss)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt_net.step()
            opt_net.zero_grad()
            opt_encoder.step()
            opt_encoder.zero_grad()

        if self.global_step % self.log_interval == 0:
            self.log_img(img, split='img')
            self.log_img(label, split='label')
            self.log_img(h, split='h')
            self.log_img(z, split='z')

        self.log_dict(net_loss_log, rank_zero_only=True, logger=True)
        self.log_dict(encoder_loss_log, rank_zero_only=True, logger=True)

    def validation_step(self, batch, batch_idx):
        img, label, cond = batch

        hs = []

        h = label.repeat(1, 3, 1, 1)
        for i, module in enumerate(self.net.encoder):
            h = module(h)
            hs.append(h)

        zs = []
        z = img
        for i, module in enumerate(self.encoder):
            z = module(z)
            zs.append(z)

        feat_losses = []
        for i, j in zip(hs, zs):
            feat_losses.append(F.mse_loss(i.detach(), j, reduction='none').mean(dim=[1, 2, 3]))
        feat_loss = sum(feat_losses).mean()

        for i, block in enumerate(self.net.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        with torch.no_grad():
            for i, block in enumerate(self.net.decoder):
                z = torch.cat([z, zs.pop()], dim=1)
                z = block(z)

        net_loss, net_loss_log = self.loss(h, label, training=True, split='net')
        encoder_loss, encoder_loss_log = self.loss(z, label, training=True, split='encoder')
        encoder_loss = encoder_loss + feat_loss
        encoder_loss_log['val/encoder/loss'] = encoder_loss
        encoder_loss_log['val/encoder/feat_loss'] = feat_loss

        if self.global_step % self.log_interval == 0:
            self.log_img(img, split='img')
            self.log_img(label, split='label')
            self.log_img(h, split='h')
            self.log_img(z, split='z')

        self.log_dict(net_loss_log, rank_zero_only=True, logger=True)
        self.log_dict(encoder_loss_log, rank_zero_only=True, logger=True)

    @torch.no_grad()
    def log_img(self, x: torch.Tensor, split='img'):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/{split}', x[0], self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.net.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    )

        opt_encoder = torch.optim.AdamW(list(self.encoder.parameters()),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay,
                                        )

        return [opt_net, opt_encoder]
