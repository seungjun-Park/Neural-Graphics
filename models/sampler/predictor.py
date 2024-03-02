import torch
import numpy as np

from .utils import Sampler
from modules.scheduler import SDESchedular, VPSDESchedular, VESDESchedular


class Predictor(Sampler):
    def __init__(self, probability_flow=False, **kwargs):
        super().__init__(**kwargs)
        self.probability_flow = probability_flow

    def initialize(self, model):
        self.model = model
        self.schedular: SDESchedular = model.schedular
        self.initialized = True

    def update(self, x, t, y=None):
        pass

    def reverse_sde(self, x, t, y=None):
        drift, diffusion = self.schedular.sde(x, t)
        score = self.model.get_score(x, t, y=None)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

    def reverse_discretize(self, x, t, y=None):
        f, G = self.schedular.discretize(x, t)
        score = self.model.get_score(x, t, y=y)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, x, t, y=None):
        assert self.initialized

        dt = -1. / self.schedular.N
        z = torch.randn_like(x)
        drift, diffusion = self.reverse_sde(x, t, y=y)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean

class ReverseDioffusionPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, x, t, y=None):
        assert self.initialized

        f, G = self.reverse_discretize(x, t, y)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class AncestralSamplingPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vesde_update(self, x, t, y=None):
        assert self.initialized

        timestep = (t * (self.schedular.N - 1) / self.schedular.T).long()
        sigma = self.schedular.sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), self.schedular.sigmas.to(t.device)[timestep - 1])
        score = self.model.get_score(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update(self, x, t, y=None):
        assert self.initialized

        timestep = (t * (self.schedular.N - 1) / self.schedular.T).long()
        beta = self.schedular.betas[timestep]
        score = self.model(x, t, y=y)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update(self, x, t, y=None):
        if isinstance(self.schedular, VESDESchedular):
            return self.vesde_update(x, t, y)
        elif isinstance(self.schedular, VPSDESchedular):
            return self.vpsde_update(x, t, y)
