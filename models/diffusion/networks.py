import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

from utils import instantiate_from_config, partial


class DiffusionWrapper(nn.Module):
    def __init__(self,
                 unet_config,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.net = instantiate_from_config(unet_config)

    @abc.abstractmethod
    def forward(self, x, y=None, cond=None, *args, **kwargs):
        pass


class DDPM(DiffusionWrapper):
    def __init__(self,
                 beta_min=1e-4,
                 beta_max=15e-3,
                 cosine_s=8e-3,
                 schedule_type='linear',
                 continuous=False,
                 N=1000,
                 eps=1e-5,
                 parameterization='eps',
                 v_posterior=0.,
                 l_simple_weight=1.0,
                 elbo_weight=0.0,
                 loss_type='l2',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s
        self.N = N
        self.continuous = continuous
        self.T = 1
        self.eps = eps
        self.l_simple_weight = l_simple_weight
        self.elbo_weight = elbo_weight
        self.v_posterior = v_posterior

        assert schedule_type.lower() in ['linear', 'cosine']
        self.schedule_type = schedule_type.lower()
        assert parameterization.lower() in ['eps', 'x0']
        self.parameterization = parameterization.lower()
        assert loss_type.lower() in ['l2', 'l1']
        self.loss_type = loss_type.lower()

        if self.schedule_type == 'linear':
            betas = torch.linspace(beta_min, beta_max, self.N)
        elif self.schedule_type == 'cosine':
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
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_1m_alphas_cumpord', to_torch(np.sqrt(1. - alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        f = -0.5 * beta_t[:, None, None, None] * x
        G = torch.sqrt(beta_t)
        return f, G

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
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
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def get_loss(self, target, predict, mean=False):
        if self.loss_type == 'l1':
            loss = (target - predict).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = F.mse_loss(target, predict)
            else:
                loss = F.mse_loss(target, predict, reduction='none')
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        return loss

    def forward(self, x, y=None, cond=None, *args, **kwargs):
        noise = torch.randn_like(x)
        if self.continuous:
            t = torch.rand((x.shape[0],), device=x.device)
            t = t * (self.T - self.eps) + self.eps
            mean, std = self.marginal_prob(x, t)
            x_noisy = mean + std[:, None, None, None] * noise
        else:
            t = torch.randint(0, self.N, (x.shape[0],), device=x.device)
            x_noisy = self.sqrt_alphas_cumprod[t, None, None, None] * x + \
                      self.sqrt_1m_alphas_cumpord[t, None, None, None] * noise

        x_pred = self.net(x_noisy, t, y=y, cond=cond)
        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x
        else:
            raise NotImplementedError(f"unknown loss type '{self.paramterization}'")

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.continuous:
            if self.elbo_weight > 0:
                g2 = self.sde(torch.zeros_like(x_pred), t)[1] ** 2
                loss = torch.square(x_pred + target / std[:, None, None, None])
                loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1) * g2
            else:
                loss = torch.square(x_pred * std[:, None, None, None] + target)
                loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = loss.mean()
            loss_dict.update({f'{prefix}/loss': loss})
        else:
            loss_simple = self.get_loss(x_pred, target, mean=False).mean(dim=[1, 2, 3])
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
            loss = self.l_simple_weight * loss_simple.mean()

            loss_vlb = self.get_loss(x_pred, target, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss += (self.elbo_weight * loss_vlb)
            loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict


class NCSN(DiffusionWrapper):
    def __init__(self,
                 sigma_min=1e-4,
                 sigma_max=15e-3,
                 continuous=False,
                 N=1000,
                 eps=1e-5,
                 parameterization='eps',
                 l_simple_weight=1.0,
                 elbo_weight=0.0,
                 loss_type='l2',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.N = N
        self.N = int(self.N)
        self.T = 1
        assert parameterization.lower() in ['eps', 'x0']
        self.parameterization = parameterization.lower()
        assert loss_type.lower() in ['l2', 'l1']
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.elbo_weight = elbo_weight
        self.continuous = continuous
        self.eps = eps

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('sigma_min', to_torch(sigma_min))
        self.register_buffer('sigma_max', to_torch(sigma_max))

        sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.register_buffer('sigmas', to_torch(torch.flip(sigmas, dims=(0, ))))

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
        logp = -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)
        return logp

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), self.sigmas[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

    def get_loss(self, target, predict, mean=False):
        if self.loss_type == 'l1':
            loss = (target - predict).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, predict)
            else:
                loss = torch.nn.functional.mse_loss(target, predict, reduction='none')
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

        return loss

    def forward(self, x, y=None, cond=None, *args, **kwargs):
        noise = torch.randn_like(x)
        if self.continuous:
            t = torch.rand((x.shape[0],), device=x.device)
            t = t * (self.T - self.eps) + self.eps
            mean, std = self.marginal_prob(x, t)
            x_noisy = mean + std[:, None, None, None] * noise
        else:
            t = torch.randint(0, self.N, (x.shape[0],), device=x.device)
            noise = noise * self.sigmas[:, None, None, None]
            x_noisy = x + noise
            noise = -noise / (self.sigmas ** 2)[:, None, None, None]

        x_pred = self.net(x_noisy, t, y=y, cond=cond)
        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x
        else:
            raise NotImplementedError(f"unknown loss type '{self.paramterization}'")

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.continuous:
            if self.elbo_weight > 0:
                g2 = self.sde(torch.zeros_like(x_pred), t)[1] ** 2
                loss = torch.square(x_pred + target / std[:, None, None, None])
                loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1) * g2
            else:
                loss = torch.square(x_pred * std[:, None, None, None] + target)
                loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = loss.mean()
        else:
            loss = self.get_loss(x_pred, target, mean=False).mean(dim=[1, 2, 3])
            loss_dict.update({f'{prefix}/loss': loss.mean()})
            loss = self.l_simple_weight * loss.mean() * self.sigmas ** 2

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict


class VPPrecond(DiffusionWrapper):
    def __init__(self,
                 beta_min=0.1,
                 beta_max=19.9,
                 eps=1e-5,
                 N=1000,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sigma_min = float(self.sigma(eps))
        self.sigma_max = float(self.sigma(1))
        self.eps = eps
        self.N = N

    def forward(self, x, y=None, cond=None, *args, **kwargs):
        t = torch.rand((x.shape[0],), device=x.device)
        sigma = self.sigma(1 + t * (self.eps - 1))[:, None, None, None]

        noise = torch.randn_like(x)
        x_noisy = x + noise * sigma

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.N - 1) * self.sigma_inv(sigma)

        F_x = self.net(x_noisy * c_in, c_noise.flatten(), y=y, cond=cond)
        D_x = c_skip * x_noisy + c_out * F_x

        weight = 1 / sigma ** 2
        loss = weight * torch.square(x.contiguous() - D_x.contiguous())

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss = torch.mean(loss, dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_max * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_max * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_max


class VEPrecond(DiffusionWrapper):
    def __init__(self,
                 sigma_min=0.02,
                 sigma_max=100,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, y=None, cond=None, *args, **kwargs):
        t = torch.rand((x.shape[0],), device=x.device)
        sigma = self.sigma(t)[:, None, None, None]

        noise = torch.randn_like(x)
        x_noisy = x + noise * sigma

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.net(x_noisy * c_in, c_noise.flatten(), y=y, cond=cond)
        D_x = c_skip * x_noisy + c_out * F_x

        weight = 1 / sigma ** 2
        loss = weight * torch.square(x.contiguous() - D_x.contiguous())

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss = torch.mean(loss, dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def sigma(self, t):
        t = torch.as_tensor(t)
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** t)


class EDMPrecond(DiffusionWrapper):
    def __init__(self,
                 p_std=1.2,
                 p_mean=-1.2,
                 sigma_data=0.5,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.p_std = p_std
        self.p_mean = p_mean
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, y=None, cond=None, *args, **kwargs):
        t = torch.randn((x.shape[0],), device=x.device)
        sigma = self.sigma(t)[:, None, None, None]

        noise = torch.randn_like(x)
        x_noisy = x + noise * sigma

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.net(x_noisy * c_in, c_noise.flatten(), y=y, cond=cond)
        D_x = c_skip * x_noisy + c_out * F_x

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = weight * torch.square(x.contiguous() - D_x.contiguous())

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss = torch.mean(loss, dim=[1, 2, 3]).mean()
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def sigma(self, t):
        t = torch.as_tensor(t)
        return (t * self.p_std + self.p_mean).exp()

