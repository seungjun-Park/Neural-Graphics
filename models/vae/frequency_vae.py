import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import pytorch_lightning as pl

from typing import List, Dict

from modules.utils import conv_nd, batch_norm_nd, activation_func
from modules.vae.down import DownBlock
from modules.vae.res_block import ResidualBlock
from modules.vae.up import UpBlock


class FrequencyVAE(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_resolutions: int,
                 out_channels: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 z_channels: int,
                 dim=2,
                 act='relu',
                 kl_weight=1e-5,
                 lr=2e-5,
                 weight_decay=0.,
                 freq_pass_eps=0.3,
                 high_freq_weight=1.0,
                 low_freq_weight=1.0,
                 log_interval=100,
                 dropout=0.1,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.dim = dim

        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.freq_pass_eps = freq_pass_eps
        self.high_freq_weight = high_freq_weight
        self.low_freq_weight = low_freq_weight
        self.log_interval = log_interval

        in_ch = in_channels
        encoder = [conv_nd(dim=dim, in_channels=in_ch, out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1)]
        in_ch = hidden_dims[0]

        for hidden_dim in hidden_dims:
            encoder += [
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    dropout=dropout,
                    act=act,
                    dim=dim
                ),
                DownBlock(
                    in_channels=hidden_dim,
                    dim=dim,
                )
            ]
            in_ch = hidden_dim

        encoder += [
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                dim=dim,
            ),
            batch_norm_nd(dim=dim, num_features=in_ch),
            activation_func(act),
            conv_nd(dim=dim, in_channels=in_ch, out_channels=z_channels, kernel_size=3, stride=1, padding=1)
        ]

        self.encoder = nn.Sequential(*encoder)

        self.in_resolutions = in_resolutions
        self.mean = conv_nd(dim=dim, in_channels=z_channels, out_channels=latent_dim, kernel_size=1)
        self.logvar = conv_nd(dim=dim, in_channels=z_channels, out_channels=latent_dim, kernel_size=1)

        decoder = list()
        hidden_dims.reverse()

        decoder += [
            conv_nd(dim=dim, in_channels=latent_dim, out_channels=z_channels, kernel_size=1),
            conv_nd(dim=dim, in_channels=z_channels, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                dim=dim,
            )
        ]

        for hidden_dim in hidden_dims:
            decoder += [
                UpBlock(
                    in_channels=in_ch,
                    dim=dim,
                ),
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    dropout=dropout,
                    act=act,
                    dim=dim,
                )
            ]
            in_ch = hidden_dim

        decoder += [
            batch_norm_nd(dim=dim, num_features=in_ch),
            activation_func(act),
            conv_nd(dim=dim, in_channels=in_ch, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, mag_and_ang):
        latent_variable = self.encoder(mag_and_ang)

        mean = self.mean(latent_variable)
        logvar = self.logvar(latent_variable)

        sample_point = self.reparameterization(mean, logvar)
        recon_mag_and_ang = self.decoder(sample_point)

        return recon_mag_and_ang, mean, logvar

    def training_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch
        freq = self.img_to_freq(img, dim=self.dim)
        mag, ang = freq.abs(), freq.angle()
        mag_and_ang = torch.cat([mag, ang], dim=1)

        recon_mag_and_ang, mean, logvar = self(mag_and_ang)
        recon_mag, recon_ang = torch.chunk(recon_mag_and_ang, 2, dim=1)
        recon_freq = recon_mag * torch.exp(1j * recon_ang)

        # loss = self.freq_loss(freq, recon_freq) + self.kl(mean, logvar)

        loss = F.mse_loss(mag_and_ang, recon_mag_and_ang) + self.kl(mean, logvar)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                prefix = 'train' if self.training else 'val'

                self.log(f'{prefix}/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                         rank_zero_only=True)
                self.log_img(img, split=f'{prefix}/img')
                recon_img = self.freq_to_img(freq=recon_freq, dim=self.dim)
                self.log_img(recon_img, split=f'{prefix}/recon')

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch
        freq = self.img_to_freq(img, dim=self.dim)
        mag, ang = freq.abs(), freq.angle()
        mag_and_ang = torch.cat([mag, ang], dim=1)

        recon_mag_and_ang, mean, logvar = self(mag_and_ang)
        recon_mag, recon_ang = torch.chunk(recon_mag_and_ang, 2, dim=1)
        recon_freq = recon_mag * torch.exp(1j * recon_ang)
        recon_img = self.freq_to_img(recon_freq)
        # loss = self.freq_loss(freq, recon_freq) + self.kl(mean, logvar)

        loss = F.mse_loss(img, recon_img) + self.kl(mean, logvar)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                prefix = 'train' if self.training else 'val'

                self.log(f'{prefix}/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True,
                         rank_zero_only=True)
                self.log_img(img, split=f'{prefix}/img')
                recon_img = self.freq_to_img(freq=recon_freq, dim=self.dim)
                self.log_img(recon_img, split=f'{prefix}/recon')

        return loss

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
            freq_r = fft.fftshift(fft.fftn(img_r, dim=dim))
            freq_g = fft.fftshift(fft.fftn(img_g, dim=dim))
            freq_b = fft.fftshift(fft.fftn(img_b, dim=dim))
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
            img_r = fft.ifftn(fft.ifftshift(freq_r), dim=dim)
            img_g = fft.ifftn(fft.ifftshift(freq_g), dim=dim)
            img_b = fft.ifftn(fft.ifftshift(freq_b), dim=dim)
            img = torch.cat([img_r, img_g, img_b], dim=1).abs()

        # for grayscale image
        elif c == 1:
            img = fft.fftshift(fft.fftn(freq, dim=dim)).abs()

        else:
            NotImplementedError(f'color channel == {c} is not supported.')

        return img

    def freq_loss(self, target, pred):
        low_pass_filter = torch.zeros(target.shape).to(target.device)
        b, c, h, w = target.shape
        half_h, half_w = h // 2, w // 2
        h_eps = int(h * self.freq_pass_eps)
        w_eps = int(w * self.freq_pass_eps)
        low_pass_filter[:, :, half_h - h_eps: half_h + h_eps, half_w - w_eps: half_w + w_eps] = 1.0

        target_low = target * low_pass_filter
        pred_low = pred * low_pass_filter
        target_low_mag, target_low_ang = target_low.abs(), target.angle()
        pred_low_mag, pred_low_ang = pred_low.abs(), pred_low.angle()

        low_freq_loss = (F.mse_loss(target_low_mag, pred_low_mag) + F.mse_loss(target_low_ang, pred_low_ang)) * self.low_freq_weight

        high_pass_filter = 1 - low_pass_filter
        target_high = target * high_pass_filter
        pred_high = pred * high_pass_filter
        target_high_mag, target_high_ang = target_high.abs(), target_high.angle()
        pred_high_mag, pred_high_ang = pred_high.abs(), pred_high.angle()

        high_freq_loss = (F.mse_loss(target_high_mag, pred_high_mag) + F.mse_loss(target_high_ang, pred_high_ang)) * self.high_freq_weight

        #return low_freq_loss + high_freq_loss

    def reparameterization(self, mean, logvar):
        x = mean + torch.sqrt(torch.exp(logvar)) * torch.randn(mean.shape).to(device=mean.device)

        return x

    def kl(self, mean, logvar):
        return 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - 1.0 - logvar, dim=[1, 2, 3]) * self.kl_weight

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
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                               list(self.mean.parameters()) +
                               list(self.logvar.parameters()) +
                               list(self.decoder.parameters()),
                               lr=self.lr,
                               weight_decay=self.weight_decay,
                               betas=(0.5, 0.99))

        return opt

