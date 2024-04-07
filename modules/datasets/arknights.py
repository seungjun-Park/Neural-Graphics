import glob
import os
import json

import cv2
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
            root = os.path.join(root, 'test')

        self.edge_names = glob.glob(f'{root}/*/edges/*.*', recursive=True)
        self.img_names = glob.glob(f'{root}/*/images/*.*', recursive=True)
        self.tag_names = glob.glob(f'{root}/*/tags/*.*', recursive=True)

    def __getitem__(self, index):
        edge_name = self.edge_names[index]
        img_name = self.img_names[index]
        tag_name = self.tag_names[index]

        img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edge = cv2.imread(f'f{edge_name}', cv2.IMREAD_GRAYSCALE)

        with open(f'{tag_name}', 'r') as f:
            tag = json.load(f)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            tag = self.target_transform(tag)

        return img, edge, tag

    def __len__(self):
        return len(self.img_names)
