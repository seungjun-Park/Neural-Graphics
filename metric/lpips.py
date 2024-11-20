import cv2
import glob

import numpy as np
import tqdm
import torch
import torchvision
import math
import os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.io import loadmat

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_score_labels = torch.tensor([0., 0., 0.])
lpips_score_content = torch.tensor([0., 0., 0.])

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

data_path = '../datasets/BSDS500/data/images/test'
# data_path = '../datasets/anime/*/*/images'
file_names = glob.glob(f'{data_path}/*.*')

for name in tqdm.tqdm(file_names):
    path, filename = name.rsplit('images', 1)
    filename, _ = filename.rsplit('.', 1)
    imgs = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    labels = cv2.imread(f'{path}/edges/{filename}.png', cv2.IMREAD_GRAYSCALE)
    uaed = cv2.imread(f'{path}/uaed/{filename}.png', cv2.IMREAD_GRAYSCALE)
    muge = cv2.imread(f'{path}/muge/{filename}.png', cv2.IMREAD_GRAYSCALE)
    sdn = cv2.imread(f'{path}/sdn/{filename}.jpg', cv2.IMREAD_GRAYSCALE)
    # labels = cv2.imread(f'{path}/edges/{filename}.png', cv2.IMREAD_GRAYSCALE)
    # uaed = cv2.imread(f'{path}/uaed/{filename}.png', cv2.IMREAD_GRAYSCALE)
    # muge = cv2.imread(f'{path}/muge/{filename}.png', cv2.IMREAD_GRAYSCALE)
    # sdn = cv2.imread(f'{path}/sdn/{filename}.png', cv2.IMREAD_GRAYSCALE)

    imgs = torchvision.transforms.transforms.ToTensor()(imgs)
    imgs = torchvision.transforms.transforms.Resize(768)(imgs)
    labels = torchvision.transforms.transforms.ToTensor()(labels)
    labels = torchvision.transforms.transforms.Resize(768)(labels)
    uaed = torchvision.transforms.transforms.ToTensor()(uaed)
    uaed = torchvision.transforms.transforms.Resize(768)(uaed)
    muge = torchvision.transforms.transforms.ToTensor()(muge)
    muge = torchvision.transforms.transforms.Resize(768)(muge)
    sdn = torchvision.transforms.transforms.ToTensor()(sdn)
    sdn = torchvision.transforms.transforms.Resize(768)(sdn)
    c, h, w = labels.shape
    if w % 8 != 0:
        w = math.ceil(w / 8) * 8
    if h % 8 != 0:
        h = math.ceil(h / 8) * 8

    imgs = torchvision.transforms.transforms.Resize([h, w])(imgs).unsqueeze(0)
    labels = torchvision.transforms.transforms.Resize([h, w])(labels).unsqueeze(0).repeat(1, 3, 1, 1)
    # labels = 1. - labels
    uaed = torchvision.transforms.transforms.Resize([h, w])(uaed).unsqueeze(0).repeat(1, 3, 1, 1)
    muge = torchvision.transforms.transforms.Resize([h, w])(muge).unsqueeze(0).repeat(1, 3, 1, 1)
    sdn = torchvision.transforms.transforms.Resize([h, w])(sdn).unsqueeze(0).repeat(1, 3, 1, 1)

    lpips_score_labels[0] += lpips(labels.to(device), uaed.to(device)).to('cpu')
    lpips_score_labels[1] += lpips(labels.to(device), muge.to(device)).to('cpu')
    lpips_score_labels[2] += lpips(labels.to(device), sdn.to(device)).to('cpu')

    lpips_score_content[0] += lpips(imgs.to(device), uaed.to(device)).to('cpu')
    lpips_score_content[1] += lpips(imgs.to(device), muge.to(device)).to('cpu')
    lpips_score_content[2] += lpips(imgs.to(device), sdn.to(device)).to('cpu')

lpips_score_labels /= len(file_names)
lpips_score_content /= len(file_names)
print(1 - lpips_score_labels)
print(1 - lpips_score_content)
