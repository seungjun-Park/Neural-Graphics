from torchvision.datasets.mnist import MNIST, FashionMNIST

from torchvision import transforms
from modules.utils import instantiate_from_config
from datasets import load_dataset


class CustomMNIST(MNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform_configs=None,
                 target_transform_configs=None,
                 channels=3,
                 download=False,
                 data_size=1000,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            transform = transforms.Compose(transform_list)

        else:
            transform = None

        if target_transform_configs is not None:
            target_transform_list = list()
            for target_transform_config in target_transform_configs['transforms']:
                target_transform_list.append(instantiate_from_config(target_transform_config))
            target_transform = transforms.Compose(target_transform_list)

        else:
            target_transform = None

        if train:
            assert data_size <= 50000
        else:
            assert data_size <= 10000

        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.channels = channels

        self.data_size = data_size

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        return img, target

    def __len__(self):
        return self.data_size


class CustomFashionMNIST(FashionMNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform_configs=None,
                 target_transform_configs=None,
                 channels=3,
                 download=False,
                 data_size=1000,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            transform = transforms.Compose(transform_list)

        else:
            transform = None

        if target_transform_configs is not None:
            target_transform_list = list()
            for target_transform_config in target_transform_configs['transforms']:
                target_transform_list.append(instantiate_from_config(target_transform_config))
            target_transform = transforms.Compose(target_transform_list)

        else:
            target_transform = None

        if train:
            assert data_size <= 50000
        else:
            assert data_size <= 10000

        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.channels = channels

        self.data_size = data_size

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        return img, target

    def __len__(self):
        return self.data_size