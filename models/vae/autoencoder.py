import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Dict

from modules.utils import conv_nd, activation_func, group_norm, instantiate_from_config
from modules.vae.down import DownBlock
from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import AttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 encoder_low_config,
                 encoder_high_config,
                 middle_block_config,
                 decoder_config,
                 z_channels,
                 latent_dim,
                 loss_config,
                 lr=2e-5,
                 weight_decay=0.,
                 log_interval=100,
                 ckpt_path=None,
                 use_fp16=False,
                 dim=2,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.use_fp16 = use_fp16
        self.automatic_optimization = False

        self.dim = dim

        self.loss = instantiate_from_config(loss_config)

        self.encoder_low = instantiate_from_config(encoder_low_config)
        self.encoder_high = instantiate_from_config(encoder_high_config)

        self.middle_block = instantiate_from_config(middle_block_config)

        self.quant_conv = conv_nd(dim=dim, in_channels=2 * z_channels, out_channels=2 * latent_dim, kernel_size=1)
        self.post_quant_conv = conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1)

        self.decoder = instantiate_from_config(decoder_config)

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

    def forward(self, low, high):
        low = self.encoder_low(low)
        high = self.encoder_high(high)

        z = torch.cat([low, high], dim=1)
        z = self.middle_block(z)

        z = self.quant_conv(z)
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.reparameterization()
        z = self.post_quant_conv(z)

        x = self.decoder(z)

        return x, posterior

    def on_train_start(self):
        self.loss.perceptual_loss.to(self.device)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        img, img_low, img_high, label = batch

        recon_img, posterior = self(img_low, img_high)

        if self.iter % self.log_interval == 0:
            prefix = 'train' if self.training else 'val'
            self.log_img(img, split=f'{prefix}/img')
            self.log_img(recon_img, split=f'{prefix}/recon')
            self.log_img(self.sample(posterior), split=f'{prefix}/sample')

        opt_ae, opt_disc = self.optimizers()

        # train encoder+decoder+logvar
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(img, recon_img, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True)

        # train the discriminator
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(img, recon_img, posterior, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True)

        self.log_ssim(img, recon_img)
        self.log_psnr(img, recon_img)

    def on_validation_start(self):
        self.loss.perceptual_loss.to(self.device)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img, img_low, img_high, label = batch

        recon_img, posterior = self(img_low, img_high)

        prefix = 'train' if self.training else 'val'
        self.log_img(img, split=f'{prefix}/img')
        self.log_img(recon_img, split=f'{prefix}/recon')
        self.log_img(self.sample(posterior), split=f'{prefix}/sample')

        aeloss, log_dict_ae = self.loss(img, recon_img, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(img, recon_img, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/total_loss", log_dict_ae["val/total_loss"])
        self.log_ssim(img, recon_img)
        self.log_psnr(img, recon_img)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    @torch.no_grad()
    def log_psnr(self, target, pred):
        prefix = 'train' if self.training else 'val'
        psnr = PeakSignalNoiseRatio()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        psnr_score = psnr(target[0], pred[0])
        self.log(f'{prefix}/psnr', psnr_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_ssim(self, target, pred):
        prefix = 'train' if self.training else 'val'
        ssim = StructuralSimilarityIndexMeasure()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        ssim_score = ssim(target[0], pred[0])
        self.log(f'{prefix}/ssim', ssim_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_img(self, img, split=''):
        tb = self.logger.experiment
        tb.add_image(f'{split}', torch.clamp(img, 0, 1)[0], self.global_step, dataformats='CHW')

    def minmax_normalize(self, x):
        max_val = torch.max(x)
        min_val = torch.min(x)

        norm_x = (x - min_val) / (max_val - min_val)

        return norm_x

    def sample(self, posterior):
        sample_point = posterior.sample()
        sample_point = self.post_quant_conv(sample_point)

        sample = self.decoder(sample_point)

        return sample

    def configure_optimizers(self):
        opt_ae = torch.optim.AdamW(list(self.encoder_low.parameters()) +
                                   list(self.encoder_high.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()),
                                   lr=self.lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=self.lr,
                                     betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.get_last_layer()


