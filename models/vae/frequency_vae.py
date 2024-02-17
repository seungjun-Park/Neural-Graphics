import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import pytorch_lightning as pl

from typing import List, Dict

from modules.utils import conv_nd, batch_norm_nd, activation_func
from modules.utils import FD, LFD, frequency_cosine_similarity
from modules.utils import img_to_freq, freq_to_img
from modules.vae.down import DownBlock
from modules.vae.up import UpBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.attn_block import MHAttnBlock
from modules.vae.distributions import DiagonalGaussianDistribution
from taming.modules.losses import LPIPS


class FrequencyVAE(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_dims: List,
                 z_channels,
                 latent_dim,
                 num_res_blocks=2,
                 dropout=0.,
                 resamp_with_conv=True,
                 num_heads=-1,
                 num_head_channels=-1,
                 act='relu',
                 dim=2,
                 lr=2e-5,
                 weight_decay=0.,
                 kl_weight=1e-5,
                 fd_weight=1e-3,
                 perceptual_weight=1.0,
                 freq_cos_sim_weight=1.0,
                 log_interval=100,
                 ckpt_path=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.perceptual_weight = perceptual_weight
        self.fd_weight = fd_weight
        self.freq_cos_sim_weight = freq_cos_sim_weight

        self.lpips = LPIPS().eval()

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

            layer.append(DownBlock(in_ch, dim=dim, use_conv=resamp_with_conv))

            if i != len(hidden_dims) - 1:
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
                batch_norm_nd(dim=dim, num_features=in_ch),
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

            layer.append(UpBlock(in_ch, dim=dim, use_conv=resamp_with_conv))

            for j in range(num_res_blocks + 1):
                layer.append(ResidualBlock(in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout=dropout,
                                           act=act,
                                           dim=dim,
                                           ))
                in_ch = out_ch

            if i != 0:
                self.up.append(nn.Sequential(*layer))

        self.up.append(
            nn.Sequential(
                batch_norm_nd(dim=dim, num_features=in_ch),
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
        z = self.post_quant_conv(z)

        for module in self.up:
            z = module(z)

        return z, posterior

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        prefix = 'train' if self.training else 'val'
        img, label = batch

        recon_img, posterior = self(img)

        perceptual_loss = self.lpips(img, recon_img)
        perceptual_loss = torch.sum(perceptual_loss) / perceptual_loss.shape[0]
        loss_dict.update({f'{prefix}/lpips_loss': perceptual_loss})

        lfd_loss = LFD(img, recon_img, dim=self.dim)
        loss_dict.update({f'{prefix}/lfd_loss': lfd_loss})

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_dict.update({f'{prefix}/kl_loss': kl_loss})

        freq_cos_sim = frequency_cosine_similarity(img, recon_img, dim=self.dim)
        loss_dict.update({f'{prefix}/freq_cos_sim': freq_cos_sim})

        loss = lfd_loss * self.fd_weight + self.kl_weight * kl_loss + self.perceptual_weight * perceptual_loss + self.freq_cos_sim_weight * freq_cos_sim

        self.log_dict(loss_dict)
        self.log(f'{prefix}/loss', loss, prog_bar=True)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, split=f'{prefix}/img')
            self.log_img(recon_img, split=f'{prefix}/recon')
            self.log_img(self.sample(posterior), split=f'{prefix}/sample')

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        prefix = 'train' if self.training else 'val'
        img, label = batch

        recon_img, posterior = self(img)

        perceptual_loss = self.lpips(img, recon_img)
        perceptual_loss = torch.sum(perceptual_loss) / perceptual_loss.shape[0]
        loss_dict.update({f'{prefix}/perceptual_loss': perceptual_loss})

        l1_loss = torch.sum((img - recon_img).abs(), dim=[1, 2, 3])
        l1_loss = torch.sum(l1_loss) / l1_loss.shape[0]

        lfd_loss = LFD(img, recon_img, dim=self.dim)
        loss_dict.update({f'{prefix}/lfd_loss': lfd_loss})

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_dict.update({f'{prefix}/kl_loss': kl_loss})

        freq_cos_sim = frequency_cosine_similarity(img, recon_img, dim=self.dim)
        loss_dict.update({f'{prefix}/freq_cos_sim': freq_cos_sim})

        loss = lfd_loss * self.fd_weight + \
               self.kl_weight * kl_loss + \
               self.perceptual_weight * perceptual_loss + \
               self.freq_cos_sim_weight * freq_cos_sim + \
               l1_loss

        self.log_dict(loss_dict)
        self.log(f'{prefix}/loss', loss, prog_bar=True)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, split=f'{prefix}/img')
            self.log_img(recon_img, split=f'{prefix}/recon')
            self.log_img(self.sample(posterior), split=f'{prefix}/sample')

        return self.log_dict

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
        opt = torch.optim.Adam(list(self.down.parameters()) +
                               list(self.quant_conv.parameters()) +
                               list(self.post_quant_conv.parameters()) +
                               list(self.up.parameters()),
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               betas=(0.5, 0.9))

        return opt


