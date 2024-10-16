import os
import argparse
import glob
import math
import os.path
import random

import cv2
import torch.cuda
import torch.nn.functional as F

import torchvision
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt

from utils import instantiate_from_config
from models.classification.transformer import SwinTransformer


class GradientNormCallback(pl.Callback):
    def __init__(self, threshold=2e5):
        super().__init__()
        self.threshold = threshold

    def on_before_zero_grad(self, trainer, pl_module, *args, **kwargs):
        grad_norms = {}
        total_norm = 0.0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5

        # Log total gradient norm
        print(f'\ngrad_norm/total: {total_norm}')


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help='path to base configs. Loaded from left-to-right. '
             'Parameters can be oeverwritten or added with command-line options of the form "--key value".',
        default=list(),
    )

    parser.add_argument(
        '--epoch',
        nargs='?',
        type=int,
        default=100,
    )

    return parser


def main():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # datamodule
    datamodule = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though

    model = instantiate_from_config(config.module)

    logger = instantiate_from_config(config.logger)

    callbacks = [instantiate_from_config(config.checkpoints[cfg]) for cfg in config.checkpoints]
    # callbacks.append(GradientNormCallback(threshold=2e5))

    trainer_configs = config.trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        # detect_anomaly=True,
        **trainer_configs
    )
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model)


def test():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # datamodule
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    device = torch.device('cuda')
    model = instantiate_from_config(config.module).eval().to(device)

    data_path = './datasets/arknights_v2/train/*/images'
    # data_path = './datasets/BIPEDv3/edges/imgs/train/rgbr/real'
    file_names = glob.glob(f'{data_path}/*.*')
    with torch.no_grad():
        for i, name in enumerate(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).to(device)
            img = torchvision.transforms.transforms.Resize([512, 512])(img)
            img = img.unsqueeze(0)
            img = model(img)
            img = img.detach().cpu()
            if len(img.shape) == 4:
                img = img[0]
            img = torchvision.transforms.ToPILImage()(img)
            p1, p2 = name.rsplit('images', 1)
            if not os.path.isdir(f'{p1}/edges_v5'):
                os.mkdir(f'{p1}/edges_v5')
            img.save(f'{p1}/edges_v5/{p2}.png', 'png')
            # p1, p2 = name.rsplit('imgs', 1)
            # img.save(f'{p1}/edge_maps/{p2}', 'png')


def classification_test():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # datamodule
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    device = torch.device('cuda')
    model = instantiate_from_config(config.module).eval().to(device).net

    data_path = '../frequency_test/1.png'
    img = cv2.imread(data_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Resize([512, 512])(img)
    img = img.unsqueeze(0).to(device)
    feat = model.embed(img)
    feats, attn_maps = [], []
    for i, module in enumerate(model.encoder):
        if i % 2 == 0:
            feat, attn_map = module(feat)
            feats.append(feat)
            attn_maps.append(attn_map)
        else:
            feat = module(feat)

    with torch.no_grad():
        for j, feat in enumerate(feats):
            for k, f in enumerate(feat[0]):
                if not os.path.isdir(f'./1/feats_{j}'):
                    os.mkdir(f'./1/feats_{j}')
                # l, _ = f.shape
                # h = int(math.sqrt(_))
                # f = f.mean(-1)
                # f = f.reshape(h, h)
                f = (f - torch.min(f)) / (torch.max(f) - torch.min(f))
                f = f.unsqueeze(0)
                f = torchvision.transforms.ToPILImage()(f)
                plt.imshow(f, cmap='inferno')
                plt.savefig(f'./1/feats_{j}/{k}.png')
                plt.close()
                # f.save(f'./0/feats_{j}/{k}.png', 'png')


def eips_test():
    parsers = get_parser()

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # datamodule
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    device = torch.device('cuda')
    model = instantiate_from_config(config.module).eval().to(device)


    img_path = './datasets/arknights_v2/train/texas/images'
    edge_path = './datasets/arknights_v2/train/texas/images'
    img_names = glob.glob(f'{img_path}/*.*')
    edge_names = glob.glob(f'{edge_path}/*.*')
    with torch.no_grad():
        for i, (img_name, edge_name) in enumerate(zip(img_names, edge_names)):
            img = cv2.imread(f'{img_name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).to(device)
            img = torchvision.transforms.transforms.Resize([512, 512])(img)
            img = img.unsqueeze(0)

            edge = cv2.imread(f'{edge_name}', cv2.IMREAD_COLOR)
            edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
            edge = torchvision.transforms.transforms.ToTensor()(edge).to(device)
            edge = torchvision.transforms.transforms.Resize([512, 512])(edge)
            edge = edge.unsqueeze(0)

            similarity = model(img, edge)[0]
            print(f'similarity: {similarity}')


if __name__ == '__main__':
    main()
    # test()
    # classification_test()
    # eips_test()
