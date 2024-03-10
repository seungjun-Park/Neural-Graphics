import torch
import pytorch_lightning as pl

from utils import instantiate_from_config, img_to_freq, freq_to_img
from modules.blocks.encoder import Encoder, ComplexEncoder
from modules.blocks.decoder import Decoder, ComplexDecoder
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 enc_dec_config,
                 middle_block_config,
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

        self.encoder = Encoder(**enc_dec_config)
        self.decoder = Decoder(**enc_dec_config)

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
        x = self.encoder(x)
        z, posterior = self.middle_block(x)
        x = self.decoder(z)

        return x, posterior

    def on_train_start(self):
        self.loss.perceptual_loss.to(self.device)

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
        img, label = batch

        recon_img, posterior = self(img)

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
    def log_img(self, img, split=''):
        tb = self.logger.experiment
        tb.add_image(f'{split}', torch.clamp(img, 0, 1)[0], self.global_step, dataformats='CHW')

    def sample(self, posterior):
        sample_point = posterior.sample()
        sample_point = self.middle_block.sampling(sample_point)
        sample = self.decoder(sample_point)

        return sample

    def configure_optimizers(self):
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters()) +
                                   list(self.middle_block.parameters()) +
                                   list(self.decoder.parameters()),
                                   lr=self.lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=self.lr,
                                     betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.get_last_layer()


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