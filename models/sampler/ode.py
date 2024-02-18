import torch
from scipy import integrate

from modules.scheduler.scheduler import SDESchedular
from .predictor import ReverseDioffusionPredictor
from utils import from_flattened_numpy, to_flattened_numpy
from .utils import Sampler


class ODESolver(Sampler):
    def __init__(self,
                 diffusion_model,
                 ):
        super().__init__()

    def initialize(self, model):
        self.model = model
        self.schedular = model.schedular
        self.initialized = True
        assert isinstance(self.schedular, SDESchedular), f'Schedular is only supported SDESchedular.'

    def denoise_update(self, x, eps, y=None):
        predictor = ReverseDioffusionPredictor(probability_flow=False)
        predictor.initialized(self.model)
        vec_eps = torch.ones_like(x.shape[0], device=x.device) * eps
        _, x = predictor.update(x, vec_eps, y=y)
        return x

    @torch.no_grad()
    def sampling(self, batch_size=1, log_interval=5, y=None, x0=None, mask=None, x_t=None, return_denoise_row=True,
                 denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, probability_flow=True, inverse_scalar=None, **kwargs):

        assert self.initialized

        shape = (batch_size, self.model.channels, self.model.image_size, self.model.image_size)
        if x_t is not None:
            x = x_t
        else:
            x = self.schedular.prior_sampling(shape).to(self.model.device)

        solution = integrate.solve_ivp(self.ode, (self.schedular.T, eps), x,
                                       rtol=rtol, atol=atol, method=method, args=(shape, y, probability_flow, x0, mask))
        nfe = solution.nfev
        x= torch.tensor(solution.y[:, -1]).reshape(shape).to(self.model.device).type(torch.float32)

        if denoise:
            x = self.denoise_update(x, eps, y=y)

        if inverse_scalar is not None:
            x = inverse_scalar(x)

        return x, nfe

    def ode(self, t, x, shape, y=None, probability_flow=True, x0=None, mask=None):
        x = from_flattened_numpy(x, shape).to(self.model.device).type(torch.float32)
        if mask is not None:
            assert x0 is not None
            x_orig = self.model.q_sample(x0, t)[0]
            x = x_orig * mask + (1. - mask) * x

        vec_t = torch.ones_like(shape[0], device=x.device) * t
        drift, diffusion = self.reverse_sde(x, vec_t, y=y, probability_flow=probability_flow)
        return to_flattened_numpy(drift)

    def reverse_sde(self, x, t, y=None, probability_flow=True):
        drift, diffusion = self.schedular.sde(x, t)
        score = self.model.get_score(x, t, y=y)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion