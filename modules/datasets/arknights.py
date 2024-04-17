import glob
import os
import json
import random

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import instantiate_from_config

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class ArknightsDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform_configs=None,
                 target_transform_configs=None,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = None

        if target_transform_configs is not None:
            target_transform_list = list()
            for target_transform_config in target_transform_configs['transforms']:
                target_transform_list.append(instantiate_from_config(target_transform_config))
            self.target_transform = transforms.Compose(target_transform_list)

        else:
            self.target_transform = None

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')
        self.tag_names = glob.glob(f'{root}/*/tags/.*.*')

        assert len(self.edge_names) == len(self.img_names) == len(self.tag_names)

    def __getitem__(self, index):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]
        tag_name = self.tag_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

        with open(f'{tag_name}', 'r') as f:
            tag = json.load(f)

        if self.transform is not None:
            img = self.transform(img)
            edge = self.transform(edge)

        if self.target_transform is not None:
            tag = self.target_transform(tag)

        tag = torch.zeros(img.shape)

        return img, edge, tag
        # return img, edge, tag

    def __len__(self):
        return len(self.img_names)


class ArknightsClassificationDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform_configs=None,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = None

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*')
        self.img_names = glob.glob(f'{root}/*/images/*.*')

        assert len(self.edge_names) == len(self.img_names)

    def __getitem__(self, index):
        tp = random.random() < 0.5
        if tp:
            edge_name = self.edge_names[index]
            label = torch.tensor([1.0])
        else:
            while True:
                new_idx = random.randrange(0, len(self))
                if new_idx != index:
                    break
            edge_name = self.edge_names[new_idx]
            label = torch.tensor([0.0])

        img_name = self.img_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge = cv2.imread(f'{edge_name}', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img = self.transform(img)
            edge = self.transform(edge)

        return img, edge, label

    def __len__(self):
        return len(self.img_names)
