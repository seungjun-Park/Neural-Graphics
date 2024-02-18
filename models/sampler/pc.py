import torch
from utils import instantiate_from_config
from .utils import Sampler
from .corrector import Corrector
from .predictor import Predictor
from tqdm import tqdm

class PCSampler(Sampler):
    def __init__(self,
                 predictor_config,
                 corrector_config,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.predictor: Predictor = instantiate_from_config(predictor_config)
        self.corrector: Corrector = instantiate_from_config(corrector_config)

    def initialize(self, model):
        self.model = model
        self.schedular = model.schedular
        self.predictor.initialize(model)
        self.corrector.initialize(model)
        self.initialized = True

    def samples(self, batch_size=1, log_interval=5, y=None, x0=None, mask=None, x_t=None, return_denoise_row=True,
                eps=1e-3, **kwargs):
        assert self.initialized

        shape = (batch_size, self.model.channels, self.model.image_size, self.model.image_size)
        device = self.schedular.betas.device

        if x_t is None:
            x = self.schedular.prior_sampling(shape).to(device)

        else:
            x = x_t.to(device)

        time_range = reversed(range(0, self.schedular.N))
        print(f"Running PC Sampling with {self.schedular.N} timesteps")

        iterator = tqdm(time_range, desc='PC Sampler', total=self.schedular.N)

        denoise_row = []
        timesteps = torch.linspace(self.schedular.T, eps, self.schedular.N, device=device)

        for i, steps in enumerate(iterator):
            if i % (self.schedular.N // log_interval) == 0:
                denoise_row.append(x)
            t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                x_orig = self.model.q_sample(x0, t)[0]
                x = x_orig * mask + (1. - mask) * x

            x, x_mean = self.corrector.update(x, t, y=y)
            x, x_mean = self.predictor.update(x, t, y=y)

        if return_denoise_row:
            return x, denoise_row

        return x