import torch
from tqdm import tqdm

from .utils import Sampler
from modules.scheduler.scheduler import VPSDESchedular


class DDIMSampler(Sampler):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, model):
        self.model = model
        self.schedular: VPSDESchedular = model.schedular
        self.initialized = True
        assert isinstance(self.schedular, VPSDESchedular)

    def samples(self, batch_size=1, ddim_steps=100, ddim_eta=1., log_interval=5, y=None, x0=None, mask=None, x_t=None,
                return_denoise_row=True, device=None, **kwargs):
        assert self.initialized

        shape = (batch_size, self.model.channels, self.model.image_size, self.model.image_size)
        device = self.schedular.betas.device if device is None else device

        if x_t is None:
            x = self.schedular.prior_sampling(shape).to(device)

        else:
            x = x_t.to(device)

        ddim_timesteps = torch.asarray(list(range(0, self.schedular.N, self.schedular.N // ddim_steps))) + 1

        time_range = torch.flip(ddim_timesteps, dims=(0,))
        total_steps = ddim_timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.schedular.alphas_cumprod[ddim_timesteps].to(device)
        alphas_prev = torch.asarray([self.schedular.alphas_cumprod[0]] + self.schedular.alphas_cumprod[ddim_timesteps[:-1]].tolist(), device=device)
        sigmas = ddim_eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        sqrt_1m_alphas = torch.sqrt(1. - alphas)
        sqrt_alphas = torch.sqrt(alphas)

        denoise_row = []

        for i, step in enumerate(iterator):
            if i % (total_steps // log_interval) == 0:
                denoise_row.append(x)
            index = total_steps - i - 1
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                x_orig = self.model.q_sample(x0, t)[0]
                x = x_orig * mask + (1. - mask) * x

            score = self.model(x, t, y=y)

            # select parameters corresponding to the currently considered timestep
            alphas_t = alphas[index, None, None, None]
            alphas_prev_t = alphas_prev[index, None, None, None]
            sigma_t = sigmas[index, None, None, None]
            sqrt_1m_alphas_t = sqrt_1m_alphas[index, None, None, None]
            sqrt_alphas_t = sqrt_alphas[index, None, None, None]

            if self.model.parameterization == 'eps':
                # current prediction for x_0
                pred_x0 = (x - sqrt_1m_alphas_t * score) / sqrt_alphas_t
                # direction pointing to x_t
                dir_xt = torch.sqrt((1. - alphas_prev_t - sigma_t ** 2)) * score
                noise = sigma_t * torch.randn_like(x)
                x = alphas_prev_t.sqrt() * pred_x0 + dir_xt + noise

            elif self.model.parameterization == 'x0':
                # current prediction for x_0
                pred_x0 = score
                score = (x - pred_x0 * sqrt_alphas_t) / sqrt_1m_alphas_t
                # direction pointing to x_t
                dir_xt = (1. - alphas_prev_t - sigma_t ** 2).sqrt() * score
                noise = sigma_t * torch.randn_like(x)
                x = alphas_prev_t.sqrt() * pred_x0 + dir_xt + noise

        if return_denoise_row:
            return x, denoise_row
        return x

