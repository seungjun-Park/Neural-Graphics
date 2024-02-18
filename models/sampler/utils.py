import abc


class Sampler(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False

    @abc.abstractmethod
    def initialize(self, model):
        pass

    def samples(self, *args, **kwargs):
        pass
