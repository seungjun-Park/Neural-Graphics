import numpy as np
import torch
import glob
import os

import cv2
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from modules.utils import instantiate_from_config


class TwoAFCDataset(Dataset):
    def __init__(self,
                 root,
                 dataset_type='train',
                 transform_configs=None,
                 ):
        if transform_configs is not None:
            transform_list = list()
            for transform_config in transform_configs['transforms']:
                transform_list.append(instantiate_from_config(transform_config))
            self.transform = transforms.Compose(transform_list)

        else:
            self.transform = None

        root = os.path.join(root, dataset_type)
        self.root = os.path.normpath(root)

        if dataset_type == 'train':
            self.subdirs = ['traditional', 'cnn', 'mix']
        elif dataset_type == 'val':
            self.subdirs = ['traditional', 'cnn']
        elif dataset_type == 'test':
            self.subdirs = ['superres', 'deblur', 'color', 'frameinterp']
        else:
            NotImplementedError(f'{dataset_type} is not exist.')

        self.file_paths = []
        for subdir in self.subdirs:
            file_path = glob.glob(f'{self.root}/{subdir}/ref/*.*')
            self.file_paths += file_path

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        root, dir_name, file_name = file_path.rsplit('/', 2)

        p0_img = cv2.imread(f'{root}/p0/{file_name}.png', cv2.IMREAD_COLOR)
        p0_img = cv2.cvtColor(p0_img, cv2.COLOR_BGR2RGB)

        p1_img = cv2.imread(f'{root}/p1/{file_name}.png', cv2.IMREAD_COLOR)
        p1_img = cv2.cvtColor(p1_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(f'{root}/ref/{file_name}.png', cv2.IMREAD_COLOR)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        judge_img = np.load(f'{root}/judge/{file_name}.npy').reshape((1, 1, 1, ))  # [0,1]
        judge_img = torch.from_numpy(judge_img)

        if self.transform is not None:
            p0_img = self.transform(p0_img)
            p1_img = self.transform(p1_img)
            ref_img = self.transform(ref_img)

        return ref_img, p0_img, p1_img, judge_img

    def __len__(self):
        return len(self.file_paths)
