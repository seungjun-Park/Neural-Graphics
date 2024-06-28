import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models.vgg import VGG
from omegaconf import DictConfig, ListConfig
from collections import namedtuple

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.sequential import AttentionSequential
from modules.blocks.attn_block import DoubleWindowCrossAttentionBlock
from modules.blocks.down import DownBlock


class EIPS(pl.LightningModule):
    def __init__(self,
                 net: VGG = None,
                 in_res: Union[int, List[int], Tuple[int]] = 64,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 dropout: float = 0.,
                 attn_mode: str = 'vanilla',
                 dim: int = 2,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 use_checkpoint: bool = True,
                 ckpt_path: str = None,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval

        self.chns = [64, 128, 256, 512, 512]
        cur_res = in_res
        self.similarity_blocks = nn.ModuleList()
        self.logit_blocks = nn.ModuleList()

        if net is None:
            self.net = vgg16()
        else:
            self.net = net.eval()
            for p in self.net.parameters():
                p.requires_grad = False

        for i, chn in enumerate(self.chns):
            self.similarity_blocks.append(
                DoubleWindowCrossAttentionBlock(
                    in_channels=chn,
                    in_res=to_2tuple(cur_res),
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    proj_bias=bias,
                    dropout=dropout,
                    use_checkpoint=use_checkpoint,
                    attn_mode=attn_mode,
                    dim=dim
                )
            )

            self.logit_blocks.append(
                conv_nd(dim, chn, 1, kernel_size=1, stride=1)
            )

            cur_res //= 2

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, img: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        out_imgs, out_edges = self.net(img), self.net(edge)
        feat_imgs, feat_edges, similarity = {}, {}, {}

        for kk in range(len(self.chns)):
            feat_imgs[kk], feat_edges[kk] = normalize_tensor(out_imgs[kk]), normalize_tensor(out_edges[kk])
            similarity[kk] = self.similarity_blocks[kk](feat_edges[kk], feat_imgs[kk])

        res = [self.logit_blocks[kk](similarity[kk]) for kk in range(len(self.chns))]
        val = torch.mean(res[0], dim=[1, 2, 3])
        for l in range(1, len(self.chns)):
            val += res[l].mean(dim=[1, 2, 3])
        val = torch.sum(val) / val.shape[0]
        return val

    def training_step(self, batch, batch_idx):
        img, edge, label = batch
        similarity = self(img, edge)
        loss = F.binary_cross_entropy(similarity, label)

        self.log('train/loss', loss, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, label = batch
        similarity = self(img, edge)
        loss = F.binary_cross_entropy(similarity, label)

        self.log('val/loss', loss, logger=True, rank_zero_only=True)

    @torch.no_grad()
    def log_img(self, img, edge):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', torch.clamp(img[0], min=0.0, max=1.0), self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge', torch.clamp(edge[0], min=0.0, max=1.0), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.logit_blocks.parameters()) +
                                list(self.similarity_blocks.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )

        return [opt]


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)
