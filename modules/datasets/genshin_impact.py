import glob
import os

import cv2
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from modules.utils import instantiate_from_config

from modules.utils import img_to_freq, freq_to_img, freq_filter

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class GenshinImpactDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 bandwidth=0.2,
                 transform_configs=None,
                 target_transform_configs=None,
                 ):
        self.bandwidth = bandwidth

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

        file_names = glob.glob(f'{root}/**/*.*', recursive=True)
        self.img_formats = set()
        self.str_formats = set()
        for i, file in enumerate(file_names):
            name, format = file.rsplit('.', 1)

            if format in IMG_FORMATS:
                self.img_formats.add(format)
            elif format in STR_FORMATS:
                self.str_formats.add(format)
            else:
                pass

            file_names[i] = name

        self.file_names = list(set(file_names))

    def __getitem__(self, index):
        file_name = self.file_names[index]

        img = cv2.imread(f'{file_name}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)

        freq = img_to_freq(img, dim=2)
        img_low = freq_to_img(freq_filter(freq, dim=2, bandwidth=[0.0, self.bandwidth]), dim=2)
        img_high = freq_to_img(freq_filter(freq, dim=2, bandwidth=[self.bandwidth, 1.0]), dim=2)

        txt_file = open(f'{file_name}.txt', 'r')
        label = txt_file.readlines()
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, img_low, img_high, label

    def __len__(self):
        return len(self.file_names)