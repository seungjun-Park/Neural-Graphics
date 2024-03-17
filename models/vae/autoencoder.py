import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union, List, Tuple

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from modules.blocks.down import DownBlock, ComplexDownBlock
from modules.blocks.res_block import ResidualBlock, ComplexResidualBlock
from modules.blocks.attn_block import AttnBlock, ComplexShiftedWindowAttnBlock
from modules.blocks.up import UpBlock
from modules.blocks.distributions import ComplexDiagonalGaussianDistribution, DiagonalGaussianDistribution
from modules.blocks.patches import PatchMerging, ComplexPatchMerging, PatchEmbedding, ComplexPatchEmbedding
from utils import instantiate_from_config, conv_nd, group_norm, to_2tuple


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: Union[List, Tuple],
                 embed_dim: int,
                 latent_dim: int,
                 loss_config=None,
                 mlp_ratio: int = 4,
                 num_blocks: int = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 groups: int = 32,
                 act: str = 'relu',
                 dim: int = 2,
                 use_conv: bool = True,
                 mode: str = 'nearest',
                 lr: float = 2e-5,
                 weight_decay: float = 0.,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 use_fp16: bool = False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.use_fp16 = use_fp16
        self.automatic_optimization = False

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.attn_dropout = attn_dropout
        self.bias = bias
        self.act = act
        self.dim = dim
        self.use_conv = use_conv

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        )

        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            up = list()

            for j in range(num_blocks):
                down.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        dim=dim,
                        groups=groups,
                    )
                )

                up.append(
                    ResidualBlock(
                        in_channels=out_ch,
                        out_channels=in_ch,
                        dropout=dropout,
                        act=act,
                        dim=dim,
                        groups=groups,
                    )
                )

                in_ch = out_ch

            if i != len(hidden_dims):
                down.append(DownBlock(in_ch, dim=dim))
                down.append(
                    AttnBlock(
                        in_ch,
                        mlp_ratio=mlp_ratio,
                        heads=num_heads,
                        num_head_channels=num_head_channels,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        groups=groups,
                    )
                )

                up.append(UpBlock(in_ch, dim=dim, mode=mode))
                up.append(
                    AttnBlock(
                        in_ch,
                        mlp_ratio=mlp_ratio,
                        heads=num_heads,
                        num_head_channels=num_head_channels,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        groups=groups,
                    )
                )

            self.encoder.append(nn.Sequential(*down))
            self.decoder.append(nn.Sequential(*(up[::-1])))

        self.encoder.append(
            nn.Sequential(
                group_norm(in_ch, groups),
                nn.Conv2d(in_ch, latent_dim * 2, kernel_size=3, stride=1, padding=1)
            )
        )

        self.decoder.append(
            nn.Sequential(
                nn.Conv2d(latent_dim, in_ch, kernel_size=3, stride=1, padding=1)
            )
        )

        self.decoder = self.decoder[::-1]

        self.quant_conv = nn.Conv2d(latent_dim * 2, latent_dim * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

        self.out = nn.Sequential(
            group_norm(embed_dim, groups),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, stride=1, padding=1)
        )

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

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
        outputs = dict()
        outputs['x'] = x
        for module in self.encoder:
            x = module(x)
        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)
        outputs['posterior'] = posterior
        x = self.post_quant_conv(posterior.reparameterization())
        for module in self.decoder:
            x = module(x)
        x = self.out(x)
        outputs['recon_x'] = x
        return outputs

    def on_train_start(self):
        self.loss.to(self.device)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        outputs = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(outputs)

        opt_ae, opt_disc = self.optimizers()

        # train encoder+decoder+logvar
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(outputs, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True)

        # train the discriminator
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(outputs, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True)

    def on_validation_start(self):
        self.loss.to(self.device)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        outputs = self(img)

        self.log_img(outputs)

        aeloss, log_dict_ae = self.loss(outputs, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(outputs, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    @torch.no_grad()
    def log_psnr(self, target, pred):
        prefix = 'train' if self.training else 'val'
        psnr = PeakSignalNoiseRatio()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        psnr_score = psnr(target, pred)
        self.log(f'{prefix}/psnr', psnr_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_ssim(self, target, pred):
        prefix = 'train' if self.training else 'val'
        ssim = StructuralSimilarityIndexMeasure()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        ssim_score = ssim(target, pred)
        self.log(f'{prefix}/ssim', ssim_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_img(self, outputs):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/x', torch.clamp(outputs['x'], 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/recon_x', torch.clamp(outputs['recon_x'], 0, 1)[0], self.global_step, dataformats='CHW')

    def sample(self, posterior):
        sample = posterior.sample()
        for module in self.decoder:
            sample = module(sample)
        sample = self.out(sample)

        return sample

    def configure_optimizers(self):
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()) +
                                   list(self.out.parameters()),
                                   lr=self.lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=self.lr,
                                     betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.out[-1].weight


class ComplexAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 enc_dec_config,
                 loss_config=None,
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

        self.encoder = ComplexEncoder(**enc_dec_config)
        self.decoder = ComplexDecoder(**enc_dec_config)

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

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
        outputs = dict()
        outputs['x'] = x
        freq = img_to_freq(x, norm='ortho')
        amp, phase = freq.abs(), freq.angle()
        posterior = self.encoder(x)
        outputs['posterior'] = posterior
        z = posterior.reparameterization()
        outputs['z'] = z
        recon_x = self.decoder(z)
        outputs['recon_x'] = recon_x

        return outputs

    def on_train_start(self):
        self.loss.perceptual_loss.to(self.device)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        outputs = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(outputs)

        opt_ae, opt_disc = self.optimizers()

        # train encoder+decoder+logvar
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(outputs, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True)

        # train the discriminator
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(outputs, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True)


    def on_validation_start(self):
        self.loss.perceptual_loss.to(self.device)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img, label = batch

        outputs = self(img)

        self.log_img(outputs)

        aeloss, log_dict_ae = self.loss(outputs, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        self.log("aeloss", aeloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True)

        discloss, log_dict_disc = self.loss(outputs, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True)

        return self.log_dict

    @torch.no_grad()
    def log_psnr(self, target, pred):
        prefix = 'train' if self.training else 'val'
        psnr = PeakSignalNoiseRatio()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        psnr_score = psnr(target, pred)
        self.log(f'{prefix}/psnr', psnr_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_ssim(self, target, pred):
        prefix = 'train' if self.training else 'val'
        ssim = StructuralSimilarityIndexMeasure()
        target = target.detach().cpu()
        pred = pred.detach().cpu()
        ssim_score = ssim(target, pred)
        self.log(f'{prefix}/ssim', ssim_score, prog_bar=False, logger=True)

        return

    @torch.no_grad()
    def log_img(self, outputs):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/x', torch.clamp(outputs['x'], 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/recon_x', torch.clamp(outputs['recon_x'], 0, 1)[0], self.global_step, dataformats='CHW')

    def sample(self, posterior):
        sample_point = posterior.sample()
        sample = self.decoder(sample_point)
        sample = freq_to_img(sample)

        return sample

    def configure_optimizers(self):
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()),
                                   lr=self.lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=self.lr,
                                     betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.get_last_layer()