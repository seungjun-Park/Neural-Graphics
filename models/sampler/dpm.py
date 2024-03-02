import torch
from tqdm import tqdm
from .utils import Sampler


class DDPMSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, model):
        self.model = model
        self.schedular: VPSDESchedular = model.schedular

        self.initialized = True
        assert isinstance(self.schedular, VPSDESchedular), f'DDPMSampler can only support VPSDEShedular.'
        assert not self.model.continuous, f'DDPMSampler can only support discrete mode.'

    def samples(self, batch_size=1, log_interval=5, y=None, x0=None, mask=None, x_t=None, return_denoise_row=True, device=None, **kwargs):
        assert self.initialized

        shape = (batch_size, self.model.channels, self.model.image_size, self.model.image_size)
        device = self.schedular.betas.device if device is None else device

        if x_t is None:
            x = self.schedular.prior_sampling(shape).to(device)

        else:
            x = x_t.to(device)

        time_range = reversed(range(0, self.schedular.N))
        total_steps = self.schedular.N
        print(f"Running DDPM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDPM Sampler', total=self.schedular.N)

        betas = self.schedular.betas.to(device)
        sqrt_betas = torch.sqrt(betas).to(device)
        alphas = 1. - betas
        sqrt_alphas = torch.sqrt(1. - betas)
        sqrt_1m_alphas_cumprod = self.schedular.sqrt_one_minus_alphas_cumprod.to(device)

        denoise_row = []

        for i, step in enumerate(iterator):
            if i % (self.schedular.N // log_interval) == 0:
                denoise_row.append(x)

            index = total_steps - i - 1
            noise = torch.randn_like(x) if index > 0 else torch.zeros_like(x)
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            if mask is not None:
                assert x0 is not None
                x_orig = self.model.q_sample(x0, t, noise)['x_t']
                x = x_orig * mask + (1. - mask) * x

            score = self.model(x, t, y=y)

            alphas_t = alphas[index, None, None, None]
            sqrt_1m_alphas_cumprod_t = sqrt_1m_alphas_cumprod[index, None, None, None]
            sqrt_betas_t = sqrt_betas[index, None, None, None]
            sqrt_alphas_t = sqrt_alphas[index, None, None, None]

            x = (x - (1. - alphas_t) / sqrt_1m_alphas_cumprod_t * score) / sqrt_alphas_t + sqrt_betas_t * noise

        if return_denoise_row:
            return x, denoise_row
        return x



