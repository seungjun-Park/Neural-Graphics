import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager
from functools import partial
from pytorch_lightning.utilities.distributed import rank_zero_only

from utils import exists, default, count_params, instantiate_from_config
from modules.ema import LitEma
from modules.diffusion.util import make_beta_schedule, extract_into_tensor
from models.diffusion.ddim import DDIMSampler

from torchmetrics.image.fid import FrechetInceptionDistance


class LatentDiffusionModel(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 vae_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 use_ema=True,
                 image_size=256,
                 channels=3,
                 log_interval=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 scheduler_config=None,
                 use_positional_encodings=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 base_learning_rate=1e-4,
                 num_classes=None,
                 ):
        super().__init__()

        self.clip_denoised = clip_denoised
        self.log_interval = log_interval
        self.learning_rate = base_learning_rate
        self.image_size = image_size
        self.channels = channels

        self.num_classes = num_classes
        self.model = instantiate_from_config(unet_config)

        self.prev_epoch = -1

        count_params(self.model, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.instantiate_first_stage(vae_config)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        # self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, cond=None, y=None, noise=None):
        output = {}

        posterior = self.encode(x=x_start, label=y)
        z_start = posterior.reparameterization()
        noise = default(noise, lambda: torch.randn_like(z_start))
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        model_output = self.model(z_noisy, t, cond, y=y)

        output['z_noisy'] = z_noisy

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_simple = self.get_loss(model_output, noise, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, noise, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, cond=None, y=None, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, cond, y, *args, **kwargs)

    def get_input(self, batch):
        x, target = batch
        x = x.to(memory_format=torch.contiguous_format).float()
        if self.num_classes is not None:
            target = target.to(memory_format=torch.contiguous_format)
        else:
            target = None

        return x, target

    @torch.no_grad()
    def decode(self, z, label=None, return_phase=False):
        recon, recon_phase = self.first_stage_model.decode(z, label, True)
        if return_phase:
            return recon, recon_phase

        return recon

    @torch.no_grad()
    def encode(self, x, label=None):
        return self.first_stage_model.encode(x, label)

    def shared_step(self, batch, **kwargs):
        x, label = self.get_input(batch)
        loss = self(x, y=label)

        return loss

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        with torch.no_grad():
            if self.prev_epoch != self.current_epoch:
                self.prev_epoch = self.current_epoch
                self.log_images(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, rank_zero_only=True)

        with torch.no_grad():
            if self.prev_epoch != self.current_epoch:
                self.prev_epoch = self.current_epoch
                self.log_images(batch)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def sample_log(self, batch_size, cond=None, y=None, ddim_steps=50, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                     shape, cond, verbose=False, y=y, **kwargs)

        return samples, intermediates

    @rank_zero_only
    @torch.no_grad()
    def log_images(self, batch, N=1, ddim_steps=100, ddim_eta=1., inpaint=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        x, target = self.get_input(batch)
        N = x.shape[0]
        log["inputs"] = x
        z = self.encode(x, target).reparameterization()
        # get denoise row
        with self.ema_scope("Plotting"):
            samples, z_denoise_row = self.sample_log(cond=None, batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta, y=target)
        x_samples, phase_samples = self.decode(samples, return_phase=True)
        log["samples"] = x_samples
        log['phase_samples'] = phase_samples

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with self.ema_scope("Plotting Inpaint"):
                samples, _ = self.sample_log(cond=None, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask, y=target)
            x_samples = self.decode(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            mask = 1. - mask
            # outpaint
            with self.ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(cond=None, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask, y=target)
            x_samples = self.decode(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        tb = self.logger.experiment

        prefix = 'train' if self.training else 'val'

        tb.add_image(f'{prefix}/input', self.normalize_image(log['inputs'][0, ...]), self.global_step)
        tb.add_image(f'{prefix}/sample_recon', self.normalize_image(log['samples'][0, ...]), self.global_step)
        tb.add_image(f'{prefix}/sample_phase', self.normalize_image(log['phase_samples'][0, ...]), self.global_step)
        tb.add_image(f'{prefix}/sample_inpainting', self.normalize_image(log['samples_inpainting'][0, ...]), self.global_step)
        tb.add_image(f'{prefix}/sample_outpainting', self.normalize_image(log['samples_outpainting'][0, ...]), self.global_step)

        # inputs = log['inputs']
        # samples = self.normalize_image(log['samples'])
        # if inputs.shape[1] != 3:
        #     inputs = inputs.repeat(1, 3, 1, 1)
        # if samples.shape[1] != 3:
        #     samples = samples.repeat(1, 3, 1, 1)
        #
        # fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        # fid.update(inputs, real=True)
        # fid.update(samples, real=False)
        # fid_score = fid.compute()
        #
        # tb.add_scalar(f'{prefix}/FID', fid_score, self.global_step)

    def normalize_image(self, x):
        max_val = torch.max(x)
        min_val = torch.min(x)

        norm_x = (x - min_val) / (max_val - min_val)

        return norm_x

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

