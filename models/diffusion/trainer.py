import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.distributed import rank_zero_only

from utils import instantiate_from_config, count_params
from modules.ema import LitEma


class DiffusionTrainer(pl.LightningModule):
    def __init__(self,
                 model_config,
                 vae_config=None,
                 image_size=256,
                 channels=3,
                 ckpt_path=None,
                 ignore_keys=[],
                 log_interval=100,
                 num_classes=None,
                 learning_rate=1e-4,
                 lr_scheudlar_config=None,
                 weight_decay=0.,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels

        self.model = instantiate_from_config(model_config)
        count_params(self.model, verbose=True)

        self.use_vae = vae_config is not None
        if self.use_vae:
            self.prepare_vae(vae_config)

        self.log_interval = log_interval
        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.use_lr_scheduler = lr_scheudlar_config is not None
        if self.use_lr_scheduler:
            self.lr_scheudlar_config = lr_scheudlar_config

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def prepare_vae(self, config):
        model = instantiate_from_config(config)
        self.vae = model.eval()
        # self.first_stage_model.train = False
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode(self, x):
        assert self.use_vae
        posterior = self.vae.encode(x)
        x = posterior.reparameterization().detach()

        return x

    @torch.no_grad()
    def decode(self, z, label=None):
        assert self.use_vae
        recon = self.vae.decode(z, label)

        return recon

    def get_input(self, batch):
        x, target = batch
        x = x.to(memory_format=torch.contiguous_format).float()
        if self.num_classes is not None:
            target = target
            target = target.to(memory_format=torch.contiguous_format)
        else:
            target = None

        return x, target

    def forward(self, batch):
        x, target = self.get_input(batch)
        if self.use_vae:
            x = self.encode(x)

        loss, loss_dict = self.model(x, y=target)

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        # with torch.no_grad():
        #     if self.global_step % self.log_interval == 0:
        #         self.log_images(batch)

        prefix = 'train' if self.training else 'val'

        self.log(f'{prefix}/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

        if self.use_lr_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        prefix = 'train' if self.training else 'val'

        self.log(f'{prefix}/loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, rank_zero_only=True)

        # with torch.no_grad():
        #     if self.global_step % self.log_interval == 0:
        #         self.log_images(batch)

    def normalize_image(self, x):
        max_val = torch.max(x)
        min_val = torch.min(x)

        norm_x = (x - min_val) / (max_val - min_val)

        return norm_x

    def normalize_phase(self, phase):
        return (phase + math.pi) / (2 * math.pi)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            assert 'target' in self.lr_scheduler_config
            scheduler = instantiate_from_config(self.lr_scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return [opt], []