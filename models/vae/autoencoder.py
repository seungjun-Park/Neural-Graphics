import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import pytorch_lightning as pl

from typing import List, Dict

from modules.utils import conv_nd, batch_norm_nd, activation_func, group_norm, instantiate_from_config
from modules.utils import img_to_freq, freq_to_img
from modules.vae.down import DownBlock
from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import MHAttnBlock
from modules.vae.fft_block import FFTAttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_dims: List,
                 z_channels,
                 latent_dim,
                 loss_config,
                 num_res_blocks=2,
                 dropout=0.,
                 resamp_with_conv=True,
                 num_heads=-1,
                 num_head_channels=-1,
                 act='relu',
                 dim=2,
                 lr=2e-5,
                 weight_decay=0.,
                 log_interval=100,
                 ckpt_path=None,
                 eps=0.,
                 use_fp16=False,
                 use_attn=True,
                 attn_type='fft',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.eps = eps
        self.use_fp16 = use_fp16
        self.use_attn = use_attn
        self.attn_type = attn_type.lower()
        self.automatic_optimization = False

        self.loss = instantiate_from_config(loss_config)

        assert num_head_channels != -1 or num_heads != 1

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.act = act
        self.dim = dim

        self.down = nn.ModuleList()

        in_ch = in_channels
        out_ch = hidden_dims[0]
        self.down.append(
            conv_nd(dim=dim,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1)
        )
        in_ch = out_ch

        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.ModuleList()
            out_ch = hidden_dim

            for j in range(num_res_blocks):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            if self.use_attn:
                if self.attn_type == 'vanilla':
                    if self.num_heads == -1:
                        heads = in_ch // self.num_head_channels
                    else:
                        heads = self.num_heads

                    layer.append(MHAttnBlock(in_channels=in_ch,
                                             heads=heads))

                elif self.attn_type == 'fft':
                    layer.append(FFTAttnBlock(in_ch))

            layer.append(DownBlock(in_ch, dim=dim, use_conv=resamp_with_conv))

            self.down.append(nn.Sequential(*layer))

        if self.num_heads == -1:
            heads = in_ch // self.num_head_channels
        else:
            heads = self.num_heads

        self.down.append(
            nn.Sequential(
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act,
                    dim=dim,
                ),
                MHAttnBlock(in_channels=in_ch,
                            heads=heads),
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    dropout=dropout,
                    act=act,
                    dim=dim,
                ),
                group_norm(in_ch),
                activation_func(act),
                conv_nd(dim=dim,
                        in_channels=in_ch,
                        out_channels=2 * z_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)
            )
        )

        self.quant_conv = conv_nd(dim=dim, in_channels=2 * z_channels, out_channels=2 * latent_dim, kernel_size=1)
        self.post_quant_conv = conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1)

        self.up = nn.ModuleList()
        self.up.append(
            conv_nd(
                dim=dim,
                in_channels=z_channels,
                out_channels=in_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        if self.num_heads == -1:
            heads = in_ch // self.num_head_channels
        else:
            heads = self.num_heads

        self.up.append(
            nn.Sequential(
                ResidualBlock(in_channels=in_ch,
                              out_channels=in_ch,
                              dropout=dropout,
                              act=act,
                              dim=dim,
                              ),
                MHAttnBlock(in_channels=in_ch,
                            heads=heads),
                ResidualBlock(in_channels=in_ch,
                              out_channels=in_ch,
                              dropout=dropout,
                              act=act,
                              dim=dim,
                              ),
            )
        )

        hidden_dims.reverse()

        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.ModuleList()
            out_ch = hidden_dim

            for j in range(num_res_blocks + 1):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            if self.use_attn:
                if self.attn_type == 'vanilla':
                    if self.num_heads == -1:
                        heads = in_ch // self.num_head_channels
                    else:
                        heads = self.num_heads

                    layer.append(MHAttnBlock(in_channels=in_ch,
                                             heads=heads))

                elif self.attn_type == 'fft':
                    layer.append(FFTAttnBlock(in_ch))

            layer.append(UpBlock(in_ch, dim=dim, use_conv=resamp_with_conv))

            self.up.append(nn.Sequential(*layer))

        self.up.append(
            nn.Sequential(
                group_norm(in_ch),
                activation_func(act),
                conv_nd(
                    dim=dim,
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

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

    def forward(self, x):
        for module in self.down:
            x = module(x)

        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)
        z = posterior.reparameterization()
        z = z
        z = self.post_quant_conv(z)

        for module in self.up:
            z = module(z)

        return z, posterior

    def training_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        recon_img, posterior = self(img)

        if self.global_step % self.log_interval == 0:
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

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # train the discriminator
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(img, recon_img, posterior, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        recon_img, posterior = self(img)

        if self.global_step % self.log_interval == 0:
            prefix = 'train' if self.training else 'val'
            self.log_img(img, split=f'{prefix}/img')
            self.log_img(recon_img, split=f'{prefix}/recon')
            self.log_img(self.sample(posterior), split=f'{prefix}/sample')

        aeloss, log_dict_ae = self.loss(img, recon_img, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(img, recon_img, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/total_loss", log_dict_ae["val/total_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict
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
        sample = self.post_quant_conv(sample_point)

        for module in self.up:
            sample = module(sample)

        return sample

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.down.parameters()) +
                                  list(self.up.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.up[-1][-1].weight


