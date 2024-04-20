import argparse
import glob
import random

import cv2
import torch.cuda
import torch.nn.functional as F

import torchvision
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from utils import instantiate_from_config


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

    checkpoint_callbacks = [instantiate_from_config(config.checkpoints[cfg]) for cfg in config.checkpoints]

    trainer_configs = config.trainer
    trainer = Trainer(logger=logger, callbacks=checkpoint_callbacks, enable_progress_bar=False, **trainer_configs)
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

    data_path = '../test/'
    file_names = glob.glob(f'{data_path}/*')
    with torch.no_grad():
        for i, name in enumerate(file_names):
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torchvision.transforms.transforms.ToTensor()(img).to(device)
            img = torchvision.transforms.transforms.Resize([1024, 1024])(img)
            img = img.unsqueeze(0)
            img = model(img)
            # img = img.detach().cpu()
            if len(img.shape) == 4:
                img = img[0]
            img = torchvision.transforms.ToPILImage()(img)
            # first, second = name.split('imgs')
            img.save(f'{data_path}_{i}.png', 'png')


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

    img_path = '../arknights_v2/val/*/images'
    edge_path = '../arknights_v2/val/*/edges'
    edges = glob.glob(f'{edge_path}/*')
    imgs = glob.glob(f'{img_path}/*')

    to_tensor = torchvision.transforms.transforms.ToTensor()
    resize = torchvision.transforms.transforms.Resize([224, 224])

    tps = []
    fps = []
    tns = []
    fns = []

    num_t = 0
    num_f = 0

    cost_fun = torch.nn.BCELoss(reduction='none')

    with torch.no_grad():
        for i, name in enumerate(imgs):
            tp = random.random() < 0.5
            if not tp:
                num_f += 1
                while True:
                    idx = random.randrange(0, len(imgs))
                    if idx != i:
                        break
            else:
                num_t += 1
                idx = i

            edge = cv2.imread(edges[idx], cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize(to_tensor(img)).to(device)
            edge = resize(to_tensor(edge)).to(device)
            img = img.unsqueeze(0)
            edge = edge.unsqueeze(0)
            logit = model(edge, img).cpu()
            prob = F.sigmoid(logit)
            cost = cost_fun(prob, torch.tensor([1. if tp else 0.]).unsqueeze(0))
            print(f'tp: {tp}', end=' ')
            print(f'logit: {logit}', end=' ')
            print(f'prob: {prob}', end=' ')
            print(f'cost: {cost}')
            if tp:
                if cost >= 1:
                    tps.append((logit, prob))
                else:
                    fns.append((logit, prob))
            else:
                if cost >= 1:
                    fps.append((logit, prob))
                else:
                    tns.append((logit, prob))

    print(f'total lens: {len(imgs)}')
    print(f'num_t: {num_t}')
    print(f'num_f: {num_f}')

    print(f'tps: {len(tps)}')
    print(f'fps: {len(fps)}')
    print(f'tns: {len(tns)}')
    print(f'fns: {len(fns)}')

    accr = (len(tps) + len(tns)) / len(imgs)
    print(f'accr: {accr}')


if __name__ == '__main__':
    # main()
    # test()
    eips_test()
