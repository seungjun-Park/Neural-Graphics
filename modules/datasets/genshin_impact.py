import glob
import os

import cv2
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from modules.utils import instantiate_from_config

IMG_FORMATS = ['png', 'jpg']
STR_FORMATS = ['txt', 'csv']


class GenshinImpactDataset(Dataset):
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

        if isinstance(root, str):
<<<<<<< HEAD
            root = os.path.join(root, self.__class__.__name__, "raw")
=======
            root = os.path.expanduser(root)
>>>>>>> parent of 529746e (modified genshin_dataset module.)

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
        for img_format in self.img_formats:
            img = cv2.imread(f'{file_name}.{img_format}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(img)

        for str_format in self.str_formats:
            txt_file = open(f'{file_name}.{str_format}', 'r')
            label = txt_file.readlines()
            if self.target_transform is not None:
                label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.file_names)