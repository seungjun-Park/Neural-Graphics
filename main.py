import argparse
import glob
import cv2

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

    model = instantiate_from_config(config.module).eval()

    data_path = '../test'
    file_names = glob.glob(f'{data_path}/*')
    for i, name in enumerate(file_names):
        img = cv2.imread(f'{name}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torchvision.transforms.transforms.ToTensor()(img)
        img = torchvision.transforms.transforms.Resize([512, 512])(img)
        img = img.unsqueeze(0)
        img = model(img)
        if len(img.shape) == 4:
            img = img[0]
        img = torchvision.transforms.ToPILImage()(img)
        img.save(f'{name}_{i}.png', 'png')


if __name__ == '__main__':
    main()
    # test()
