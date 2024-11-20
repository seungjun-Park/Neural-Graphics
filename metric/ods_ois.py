import numpy as np
import torch
import glob
import cv2
import tqdm
import torchvision
from torchmetrics.classification import BinaryF1Score
import math
from scipy.io import loadmat

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_fscore(pred, gt, threshold=0.5):
    pred = (pred > threshold).float()
    gt = (gt > 0.5).float()

    tp = (pred * gt)
    fp = (pred * (1 - gt))
    fn = ((1 - pred) * gt)
    num_tp = torch.sum(tp == 0)
    num_fp = torch.sum(fp == 0)
    num_fn = torch.sum(fn == 1)

    precision = num_tp / (num_tp + num_fp + 1e-8)
    recall = num_tp / (num_tp + num_fn + 1e-8)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), fscore.item()


thresholds = np.linspace(0.01, 0.99, 100)

ods_scores = [{t: [] for t in thresholds}, {t: [] for t in thresholds}, {t: [] for t in thresholds}]
ois_scores = [[], [], []]

# data_path = '../datasets/BSDS500/data/images/test'
data_path = '../datasets/anime/*/*/images'
file_names = glob.glob(f'{data_path}/*.*')

for name in tqdm.tqdm(file_names):
    path, filename = name.rsplit('images', 1)
    filename, _ = filename.rsplit('.', 1)
    labels = cv2.imread(f'{path}/edges/{filename}.png', cv2.IMREAD_GRAYSCALE)
    uaed = cv2.imread(f'{path}/uaed/{filename}.png', cv2.IMREAD_GRAYSCALE)[1:, 1:]
    muge = cv2.imread(f'{path}/muge/{filename}.png', cv2.IMREAD_GRAYSCALE)[1:, 1:]
    sdn = cv2.imread(f'{path}/sdn/{filename}.png', cv2.IMREAD_GRAYSCALE)

    labels = torchvision.transforms.transforms.ToTensor()(labels)
    uaed = torchvision.transforms.transforms.ToTensor()(uaed)
    muge = torchvision.transforms.transforms.ToTensor()(muge)
    sdn = torchvision.transforms.transforms.ToTensor()(sdn)
    c, h, w = labels.shape

    labels = torchvision.transforms.transforms.Resize([h, w])(labels)
    uaed = torchvision.transforms.transforms.Resize([h, w])(uaed)
    muge = torchvision.transforms.transforms.Resize([h, w])(muge)
    sdn = torchvision.transforms.transforms.Resize([h, w])(sdn)

    uaed = 1 - uaed
    muge = 1 - muge
    sdn = 1 - sdn

    # tp = labels * uaed
    # fp = (1. - labels) * uaed
    # fn = labels * (1. - uaed)
    # num_tp = tp.sum()
    # num_fp = fp.sum()
    # num_fn = fn.sum()
    # precision = num_tp / (num_tp + num_fp)
    # recall = num_tp / (num_tp + num_fn)
    # f1 = (2 * precision * recall) / (precision + recall)
    # print(f'num_tp: {num_tp}, num_fp: {num_fp}, precision: {precision}, recall: {recall}, F1: {f1}')
    # torchvision.transforms.ToPILImage()(tp.float()).save(f'{path}/tp/{filename}.png', 'png')
    # torchvision.transforms.ToPILImage()(fp.float()).save(f'{path}/fp/{filename}.png', 'png')
    # torchvision.transforms.ToPILImage()(fn.float()).save(f'{path}/fn/{filename}.png', 'png')
    image_fscores = [{}, {}, {}]

    for t in thresholds:
        labels = 1 - labels.long()
        bf1_score = BinaryF1Score(threshold=t).to(device)

        fscore = bf1_score(uaed.to(device), labels.to(device))
        ods_scores[0][t].append(fscore.item())
        image_fscores[0][t] = fscore.item()

        fscore = bf1_score(muge.to(device), labels.to(device))
        ods_scores[1][t].append(fscore.item())
        image_fscores[1][t] = fscore.item()

        fscore = bf1_score(sdn.to(device), labels.to(device))
        ods_scores[2][t].append(fscore.item())
        image_fscores[2][t] = fscore.item()

    best_thresh = max(image_fscores[0], key=image_fscores[0].get)
    ois_scores[0].append(image_fscores[0][best_thresh])

    best_thresh = max(image_fscores[1], key=image_fscores[1].get)
    ois_scores[1].append(image_fscores[1][best_thresh])

    best_thresh = max(image_fscores[2], key=image_fscores[2].get)
    ois_scores[2].append(image_fscores[2][best_thresh])

# Compute ODS
average_fscores = {thresh: np.mean(scores) for thresh, scores in ods_scores[0].items()}
ods_thresh = max(average_fscores, key=average_fscores.get)
ods_scores[0] = average_fscores[ods_thresh]

average_fscores = {thresh: np.mean(scores) for thresh, scores in ods_scores[1].items()}
ods_thresh = max(average_fscores, key=average_fscores.get)
ods_scores[1] = average_fscores[ods_thresh]

average_fscores = {thresh: np.mean(scores) for thresh, scores in ods_scores[2].items()}
ods_thresh = max(average_fscores, key=average_fscores.get)
ods_scores[2] = average_fscores[ods_thresh]

# Compute OIS
ois_scores[0] = np.mean(ois_scores[0])
ois_scores[1] = np.mean(ois_scores[1])
ois_scores[2] = np.mean(ois_scores[2])

print(ods_scores)
print(ois_scores)

