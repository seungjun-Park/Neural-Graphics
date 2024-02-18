import abc
import torch

from .utils import Sampler
from modules.scheduler.scheduler import SDESchedular, VPSDESchedular

class Corrector(Sampler):
    def __init__(self, snr=0.16, n_steps=1, **kwargs):
        super().__init__()
        self.snr = snr
        self.n_steps = n_steps

    def initialize(self, model):
        self.model = model
        self.schedular: SDESchedular = model.schedular
        self.initialized =True
        assert isinstance(self.schedular, SDESchedular)

    @abc.abstractmethod
    def update(self, x, t, y=None):
        pass


class LangevinCorrector(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, x, t, y=None):
        assert self.initialized

        if isinstance(self.schedular, VPSDESchedular):
            timestep = (t * (self.schedular.N -1) / self.schedular.T).long()
            alpha = self.schedular.alphas[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(self.n_steps):
            grad = self.model(x, t, y=y)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, x, t, y=None):
        assert self.initialized

        if isinstance(self.schedular, VPSDESchedular):
            timestep = (t * (self.schedular.N -1) / self.schedular.T).long()
            alpha = self.schedular.alphas[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.schedular.marginal_prob(x, t)[1]

        for i in range(self.n_steps):
            grad = self.model(x, t, y=y)
            noise = torch.randn_like(x)
            step_size = (self.snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean
