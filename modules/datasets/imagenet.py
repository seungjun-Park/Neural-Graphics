import glob
import os
import json
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from torchvision.datasets import ImageNet
from omegaconf import DictConfig
from typing import Union, List, Tuple, Any

from utils import instantiate_from_config, to_2tuple


class ImageNetWrapper(ImageNet):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform_configs: DictConfig = None,
                 target_transform_configs: DictConfig = None,
                 **kwargs: Any
                 ):
        super().__init__(root=root, split=split, **kwargs)

        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        if target_transform_configs is not None:
            target_transform_list = list()
            for target_transform_config in target_transform_configs['transforms']:
                target_transform_list.append(instantiate_from_config(target_transform_config))
            self.target_transform = transforms.Compose(target_transform_list)


