import abc
import torch
import torch.nn as nn
import numpy as np
from functools import partial


class Scheduler(abc.ABC):
    def __init__(self):
        super().__init__()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    @abc.abstractmethod
    def make_schedule(self, *args, **kwargs):
        pass


class SDEScheduler(Scheduler):
    def __init__(self,
                 N=1000,
                 device=None
                 ):
        super().__init__()
        self.N = N
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    @abc.abstractmethod
    def make_schedule(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))

        return f, G


class VPScheduler(SDEScheduler):
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 cosine_s=8e-3,
                 type='linear',
                 v_posterior=0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.v_posterior = v_posterior

        self.make_schedule(beta_min=beta_min, beta_max=beta_max, cosine_s=cosine_s, type=type)

    def make_schedule(self, beta_min=0.1, beta_max=20, cosine_s=8e-3, type='linear', *args, **kwargs):
        type = type.lower()
        if type == 'linear':
            betas = (torch.linspace(beta_min / self.N, beta_max / self.N, self.N))
        elif type == 'cosine':
            self.N = torch.arange(self.N + 1, dtype=torch.float64) / self.N + cosine_s
            alphas = self.N / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas, a_min=0, a_max=0.999)

        else:
            raise NotImplementedError(f'{type} is not supported yet.')

        betas = betas.numpy()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.N = int(self.N)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('beta_0', to_torch(beta_min))
        self.register_buffer('beta_1', to_torch(beta_max))

        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphase_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

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

        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights)
        assert not torch.isnan(self.lvlb_weights).all()

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0.to(x.device) + t * (self.beta_1.to(x.device) - self.beta_0.to(x.device))
        f = -0.5 * beta_t[:, None, None, None] * x
        G = torch.sqrt(beta_t)
        return f, G

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1.to(x.device) - self.beta_0.to(x.device)) - 0.5 * t * self.beta_0.to(x.device)
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        timestep = (t * (self.N - 1)/ self.T).long()
        beta = self.betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class VEScheduler(SDEScheduler):
    def __init__(self,
                 sigma_min=0.01,
                 sigma_max=50,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.make_schedule(sigma_min=sigma_min, sigma_max=sigma_max)

    def make_schedule(self, sigma_min=0.01, sigma_max=50, *args, **kwargs):
        self.N = int(self.N)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('sigma_min', to_torch(sigma_min))
        self.register_buffer('sigma_max', to_torch(sigma_max))

        sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.register_buffer('sigmas', to_torch(torch.flip(sigmas, dims=(0, ))))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp = -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
        return logp

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class EDMScheduler(SDEScheduler):
    def __init__(self,
                 p_mean=-1.2,
                 p_std=1.2,
                 sigma_data=0.5,
                 sigma_min=0.1,
                 sigma_max=50,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.make_schedule(p_mean=p_mean, p_std=p_std, sigma_data=sigma_data, sigma_min=sigma_min, sigma_max=sigma_max)

    def make_schedule(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5, sigma_min=0.01, sigma_max=50, *args, **kwargs):
        self.N = int(self.N)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('sigma_min', to_torch(sigma_min))
        self.register_buffer('sigma_max', to_torch(sigma_max))
        self.register_buffer('p_mean', to_torch(p_mean))
        self.register_buffer('p_std', to_torch(p_std))
        self.register_buffer('sigma_data', to_torch(sigma_data))

        sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.register_buffer('sigmas', to_torch(torch.flip(sigmas, dims=(0, ))))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp = -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
        return logp

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G