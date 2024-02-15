import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import pytorch_lightning as pl

from typing import List, Dict

from modules.utils import conv_nd, batch_norm_nd, activation_func
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
                 kl_weight=1e-5,
                 lr=2e-5,
                 weight_decay=0.,
                 freq_pass_eps=0.3,
                 high_freq_weight=1.0,
                 low_freq_weight=1.0,
                 perceptual_weight=1.0,
                 log_interval=100,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.freq_pass_eps = freq_pass_eps
        self.high_freq_weight = high_freq_weight
        self.low_freq_weight = low_freq_weight
        self.log_interval = log_interval
        self.perceptual_weight = perceptual_weight

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

        self.ang_out = nn.Hardtanh(-math.pi, math.pi)
        self.mag_out = nn.ReLU()

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
        img, label = batch

        recon_img, posterior = self(img)

        freq_loss = self.freq_loss(img, recon_img, loss_dict)

        # recon_loss = self.lpips(img, recon_img)
        # recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        # loss_dict.update({'train/recon_loss': recon_loss})

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_dict.update({'train/kl_loss': kl_loss})

        loss = freq_loss * self.perceptual_weight + self.kl_weight * kl_loss

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                prefix = 'train' if self.training else 'val'

                self.log_dict(loss_dict)
                self.log(f'{prefix}/loss', loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.log_img(img, split=f'{prefix}/img')
                self.log_img(self.minmax_normalize(recon_img), split=f'{prefix}/recon')

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        img, label = batch

        recon_img, posterior = self(img)

        freq_loss = self.freq_loss(img, recon_img, loss_dict)

        # recon_loss = self.lpips(img, recon_img)
        # recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        # loss_dict.update({'val/recon_loss': recon_loss})


        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_dict.update({'val/kl_loss': kl_loss})

        loss = freq_loss + self.kl_weight * kl_loss

        #loss = recon_loss * self.perceptual_weight + self.kl_weight * kl_loss

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                self.log_dict(loss_dict)
                self.log(f'val/loss', loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.log_img(img, split=f'val/img')
                self.log_img(self.minmax_normalize(recon_img), split=f'val/recon')

    def img_to_freq(self, img, dim=2):
        b, c, h, w = img.shape
        if dim == 1:
            dim = [-1]
        elif dim == 2:
            dim = [-2, -1]
        else:
            dim = [-3, -2, -1]

        # for color image
        if c == 3:
            img_r, img_g, img_b = torch.chunk(img, 3, dim=1)
            freq_r = fft.fftshift(fft.fftn(img_r, dim=dim, norm='ortho'))
            freq_g = fft.fftshift(fft.fftn(img_g, dim=dim, norm='ortho'))
            freq_b = fft.fftshift(fft.fftn(img_b, dim=dim, norm='ortho'))
            freq = torch.cat([freq_r, freq_g, freq_b], dim=1)

        # for grayscale image
        elif c == 1:
            freq = fft.fftshift(fft.fftn(img, dim=dim))

        else:
            NotImplementedError(f'color channel == {c} is not supported.')

        return freq

    def freq_to_img(self, freq, dim=2):
        b, c, h, w = freq.shape
        if dim == 1:
            dim = [-1]
        elif dim == 2:
            dim = [-2, -1]
        else:
            dim = [-3, -2, -1]

        # for color image
        if c == 3:
            freq_r, freq_g, freq_b = torch.chunk(freq, 3, dim=1)
            img_r = fft.ifftn(fft.ifftshift(freq_r), dim=dim, norm='ortho')
            img_g = fft.ifftn(fft.ifftshift(freq_g), dim=dim, norm='ortho')
            img_b = fft.ifftn(fft.ifftshift(freq_b), dim=dim, norm='ortho')
            img = torch.cat([img_r, img_g, img_b], dim=1).abs()

        # for grayscale image
        elif c == 1:
            img = fft.ifftn(fft.ifftshift(freq), dim=dim, norm='ortho').abs()

        else:
            NotImplementedError(f'color channel == {c} is not supported.')

        return img

    def freq_loss(self, target, pred, loss_dict=None):
        target_freq = self.img_to_freq(target, dim=self.dim)
        pred_freq = self.img_to_freq(pred, dim=self.dim)

        low_pass_filter = torch.zeros(target_freq.shape).to(target_freq.device)
        b, c, h, w = target.shape
        half_h, half_w = h // 2, w // 2
        h_eps = int(h * self.freq_pass_eps)
        w_eps = int(w * self.freq_pass_eps)
        low_pass_filter[:, :, half_h - h_eps: half_h + h_eps, half_w - w_eps: half_w + w_eps] = 1.0

        target_freq_low = target_freq * low_pass_filter
        pred_freq_low = pred_freq * low_pass_filter

        recon_target_low = self.freq_to_img(target_freq_low, dim=self.dim)
        recon_pred_low = self.freq_to_img(pred_freq_low, dim=self.dim)

        low_freq_loss = self.lpips(recon_target_low, recon_pred_low)
        low_freq_loss = torch.sum(low_freq_loss) / low_freq_loss.shape[0]

        high_pass_filter = 1 - low_pass_filter
        target_freq_high = target_freq * high_pass_filter
        pred_freq_high = pred_freq * high_pass_filter

        recon_target_high = self.freq_to_img(target_freq_high, dim=self.dim)
        recon_pred_high = self.freq_to_img(pred_freq_high, dim=self.dim)

        high_freq_loss = self.lpips(recon_target_high, recon_pred_high)
        high_freq_loss = torch.sum(high_freq_loss) / high_freq_loss.shape[0]

        if loss_dict is not None:
            prefix = 'train' if self.training else 'val'
            loss_dict.update({f'{prefix}/low_freq_loss': low_freq_loss})
            loss_dict.update({f'{prefix}/high_freq_loss': high_freq_loss})

        return self.low_freq_weight * low_freq_loss + self.high_freq_weight * high_freq_loss

    def log_img(self, img, split=''):
        tb = self.logger.experiment
        tb.add_image(f'{split}', self.minmax_normalize(img)[0], self.global_step)

    def minmax_normalize(self, x):
        max_val = torch.max(x)
        min_val = torch.min(x)

        norm_x = (x - min_val) / (max_val - min_val)

        return norm_x

    def sample(self, shape):
        sample_point = torch.randn(shape)
        sample_mag_and_ang, _, _ = self.decoder(sample_point)
        mag, ang = torch.chunk(sample_mag_and_ang, 2, dim=1)
        frequency = torch.abs(mag) * torch.exp(ang * -1j)
        sample_img = self.freq_to_img(frequency, dim=self.dim)

        return sample_img

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.down.parameters()) +
                               list(self.quant_conv.parameters()) +
                               list(self.post_quant_conv.parameters()) +
                               list(self.up.parameters()),
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               betas=(0.5, 0.9))

        return opt

