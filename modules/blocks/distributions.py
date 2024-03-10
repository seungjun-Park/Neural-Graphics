import math

import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiagonalGaussianDistribution(object):
    def __init__(self,
                 parameters,
                 deterministic=False,
                 eps=25):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -eps, eps)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        x = torch.randn_like(x)
        return x

    def reparameterization(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class ComplexDiagonalGaussianDistribution(object):
    def __init__(self,
                 parameters,
                 deterministic=False,
                 eps=25):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        log_var_real = torch.clamp(self.logvar.real, -eps, eps)
        log_var_imag = torch.clamp(self.logvar.imag, -eps, eps)
        self.logvar = torch.complex(log_var_real, log_var_imag)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x_real = torch.randn(self.mean.shape).to(device=self.mean.device)
        x_imag = torch.randn(self.mean.shape).to(device=self.mean.device)
        return torch.complex(x_real, x_imag)

    def reparameterization(self):
        x_real = self.mean.real + self.std.real * torch.randn(self.mean.shape).to(device=self.parameters.device)
        x_imag = self.mean.imag + self.std.imag * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return torch.complex(x_real, x_imag)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                real_kl = self._kl(self.mean.real, self.var.real, self.logvar.real)
                imag_kl = self._kl(self.mean.imag, self.var.imag, self.logvar.imag)
                return real_kl + imag_kl

    def _kl(self, mean, var, logvar):
        return 0.5 * torch.sum((torch.pow(mean, 2) + var) - 1.0 - logvar, dim=[1, 2, 3])