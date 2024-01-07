import argparse
import math
import os
import time

import numpy as np
import torch
import torchvision

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from utils.util import instantiate_from_config
import utils.hologram.h_utils as h_utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

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

    parser.add_argument(
        '--batch_size',
        nargs='?',
        type=int,
        default=1,
    )

    return parser


def main():
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

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
    # trainer_configs['logger'] = logger
    # trainer_configs['callbacks'] = checkpoint_callbacks
    # trainer = Trainer(accelerator='gpu', max_epochs=opt.epoch, logger=logger, callbacks=checkpoint_callbacks)
    trainer = Trainer(logger=logger, callbacks=checkpoint_callbacks, **trainer_configs)
    trainer.fit(model=model, datamodule=datamodule)

def holonet_test():
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    convertTensor2PIL = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    model = instantiate_from_config(config.module)
    model.eval()
    model.to(device)

    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule._train_dataloader()

    for i, batch in enumerate(train_dataloader):
        img, label = model.get_input(batch)
        img = img.to(device)
        label = label.to(device)

        start = time.time()
        phase = model.generate_phase(img)
        poh = torch.ones_like(phase) * torch.exp(phase * 1j)
        recon = model.backward_propagation(poh)
        recon = torch.sum(recon.abs()**2 * model.scale, dim=1, keepdim=True)
        recon = torch.pow(recon, 0.5)
        recon = h_utils.crop_image(recon, model.homography_res, stacked_complex=False)
        end = time.time()

        if i < 100:
            sample = recon.cpu()
            sample = convertTensor2PIL(sample[0])
            sample.save(f'./test/sample_{i}.png', 'png')
            phase = (phase + math.pi) / (2 * math.pi)
            phase = phase.cpu()
            phase = convertTensor2PIL(phase[0])
            phase.save(f'./test/phase_{i}.png', 'png')

    # trainer = Trainer(accelerator='gpu', max_epochs=opt.epoch, logger=logger, callbacks=checkpoint_callbacks)
    # trainer.fit(model=model, datamodule=datamodule)

def diffusion_sampling(path):
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    convertTensor2PIL = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    model = instantiate_from_config(config.module)
    model.eval()
    model.to(device)

    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule._train_dataloader()

    if not os.path.isdir(path):
        os.makedirs(path)

    for i, batch in enumerate(train_dataloader):
        img, target = model.get_input(batch)

        samples = model.sample_log(ddim_steps=50)
        samples = model.normalize_image(samples)

        if i < 100:
            img = img.cpu().detach()
            img = convertTensor2PIL(img[0])
            img.save(f'{path}/gt_{i}.png', 'png')

            samples = samples.cpu().detach()
            samples = convertTensor2PIL(samples[0])
            samples.save(f'{path}/samples_{i}.png', 'png')
            # samples_phase = samples_phase.cpu().detach()
            # samples_phase = convertTensor2PIL(samples_phase[0])
            # samples_phase.save(f'{path}/samples_phase_{i}.png', 'png')


def ldm_sampling():
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    convertTensor2PIL = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    model = instantiate_from_config(config.module)
    model.eval()
    model.to(device)

    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule._train_dataloader()

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for i, batch in enumerate(train_dataloader):
        img, label = model.get_input(batch)
        img = img.to(device)
        label = label.to(device)

        posterior = model.encode(img, label)
        z = posterior.reparameterization()
        N, c, h, w = z.shape
        # make a simple center square
        mask = torch.ones(N, h, w).to(device)
        # zeros will be filled in
        mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
        mask = mask[:, None, ...]

        inpaints, _ = model.sample_log(batch_size=N, y=label, ddim_steps=100, x0=z[:N], mask=mask)
        inpaints, inpaints_phases = model.decode(inpaints, return_phase=True)
        inpaints_phases = model.normalize_image(inpaints_phases)
        inpaints = model.normalize_image(inpaints)
        mask = 1 - mask
        outpaints, _ = model.sample_log(batch_size=N, y=label, ddim_steps=100, x0=z[:N], mask=mask)
        outpaints, outpaints_phasese = model.decode(outpaints, return_phase=True)
        outpaints_phasese = model.normalize_image(outpaints_phasese)
        outpaints = model.normalize_image(outpaints)

        t = torch.randint(model.num_timesteps//2, model.num_timesteps//2 + 1, (z.shape[0],), device=device).long()
        z_t = model.q_sample(z, t)

        recons, _ = model.sample_log(batch_size=N, y=label, ddim_steps=100, x_T=z_t)
        recons, phases = model.decode(recons, return_phase=True)
        recons = model.normalize_image(recons)
        phases = model.normalize_image(phases)

        samples, intermediates = model.sample_log(batch_size=img.shape[0], y=label, ddim_steps=100)
        samples_recon, samples_phase = model.decode(samples, return_phase=True)
        samples_recon = model.normalize_image(samples_recon)
        samples_phase = model.normalize_image(samples_phase)

        if i < 100:
            print(f'inpaint psnr: {psnr(img, inpaints)}')
            print(f'inpaint ssim: {ssim(img, inpaints)}')
            print(f'\noutpaint psnr: {psnr(img, outpaints)}')
            print(f'outpaint ssim: {ssim(img, outpaints)}')
            print(f'\nrecons psnr: {psnr(img, recons)}')
            print(f'recons ssim: {ssim(img, recons)}')

            img = img.cpu()
            img = convertTensor2PIL(img[0])
            img.save(f'./images/ldm/fashion_mnist/inputs/img_{i}_{label[0]}.png', 'png')

            inpaints = inpaints.cpu()
            inpaints = convertTensor2PIL(inpaints[0])
            inpaints.save(f'./images/ldm/fashion_mnist/inpaints/inpaints_{i}_{label[0]}.png', 'png')
            inpaints_phases = inpaints_phases.cpu()
            inpaints_phases = convertTensor2PIL(inpaints_phases[0])
            inpaints_phases.save(f'./images/ldm/fashion_mnist/inpaints/inpaints_phases_{i}_{label[0]}.png', 'png')

            outpaints = outpaints.cpu()
            outpaints = convertTensor2PIL(outpaints[0])
            outpaints.save(f'./images/ldm/fashion_mnist/outpaints/outpaints_{i}_{label[0]}.png', 'png')
            outpaints_phasese = outpaints_phasese.cpu()
            outpaints_phasese = convertTensor2PIL(outpaints_phasese[0])
            outpaints_phasese.save(f'./images/ldm/fashion_mnist/outpaints/outpaints_phasese_{i}_{label[0]}.png', 'png')

            recons = recons.cpu()
            recons = convertTensor2PIL(recons[0])
            recons.save(f'./images/ldm/fashion_mnist/recons/recons_{i}_{label[0]}.png', 'png')
            phases = phases.cpu()
            phases = convertTensor2PIL(phases[0])
            phases.save(f'./images/ldm/fashion_mnist/recons/recons_phases_{i}_{label[0]}.png', 'png')

            samples_recon = samples_recon.cpu()
            samples_recon = convertTensor2PIL(samples_recon[0])
            samples_recon.save(f'./images/ldm/fashion_mnist/samples/samples_{i}_{label[0]}.png', 'png')
            samples_phase = samples_phase.cpu()
            samples_phase = convertTensor2PIL(samples_phase[0])
            samples_phase.save(f'./images/ldm/fashion_mnist/samples/samples_phase_{i}_{label[0]}.png', 'png')

def visualization_latent_space(path):
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    convertTensor2PIL = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    model = instantiate_from_config(config.module)
    model.eval()
    model.to(device)

    datamodule.prepare_data()
    datamodule.setup()
#    datamodule.to(device)

    val_dataloader = datamodule._validation_dataloader()

    labels = []
    zs = []

    # fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    # reals = []
    # fakes = []

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    cnt = 0
    ssim_score = 0
    psnr_score = 0

    fid_cnt = 0
    fid_score = 0

    if not os.path.isdir(path):
        os.makedirs(path)

    for i, batch in enumerate(val_dataloader):
        img, label = model.get_input(batch)
        img = img.to(device)
        label = label.to(device)
        outputs = model(img, label)

        posterior = outputs['posterior']
        recon = outputs['recon']

        sample = model.sample(posterior, label)
        sample = model.normalize_image(sample)

        recon = recon.cpu().detach()
        img = img.cpu().detach()
        sample = sample.cpu().detach()

        psnr_score += psnr(img, recon)
        ssim_score += ssim(img, recon)

        if i < 100:
            sample = convertTensor2PIL(sample[0])
            sample.save(f'{path}/sample_{i}.png', 'png')

        print(i)

        mean = posterior.mean
        mean = mean.reshape(mean.shape[0], -1)

        zs.append(mean.cpu().detach())
        labels.append(label.cpu().detach())

        cnt += 1
        # if i % 10 == 0 and i != 0:
        #     rs = torch.cat(reals, dim=0)
        #     fs = torch.cat(fakes, dim=0)
        #
        #     rs = rs
        #     fs = fs
        #
        #     if rs.shape[1] != 3:
        #         rs = rs.repeat(1, 3, 1, 1)
        #         fs = fs.repeat(1, 3, 1, 1)
        #
        #     fid.update(rs, real=True)
        #     fid.update(fs, real=False)
        #     score = fid.compute()
        #
        #     reals = []
        #     fakes = []
        #
        #     fid_score += score
        #     fid_cnt +=1

    # fid_score /= fid_cnt
    # print(f'fid_score: {fid_score}')

    psnr_score /= cnt
    ssim_score /= cnt

    print(f'psnr: {psnr_score}')
    print(f'ssim: {ssim_score}')

    zs = torch.cat(labels, dim=0)
    labels = torch.cat(labels, dim=0)

    visualization = TSNE(n_components=2).fit_transform(zs)

    plt.figure(figsize=(10, 10))
    for i, target in enumerate(set(labels.numpy())):
        indices = (labels == target)
        plt.scatter(visualization[indices, 0], visualization[indices, 1], label=str(target))

    # zs = np.array(zs)
    #
    # # Creating a scatter plot
    # fig, ax = plt.subplots(figsize=(10, 10))
    # scatter = ax.scatter(x=zs[:, 0], y=zs[:, 1], s=2.0,
    #                      c=labels, cmap='tab10', alpha=0.9, zorder=2)
    #
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    #
    # ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
    # plt.show()

    plt.legend()
    plt.savefig(f'{path}/latent_space.png')


path = 'sample/ldm/mnist/default'

if __name__ == '__main__':
    main()
    # diffusion_sampling(path)
    # holonet_test()
    # ldm_sampling()
    # visualization_latent_space('sample/mnist/default')
