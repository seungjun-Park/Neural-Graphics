import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.models.layers import DropPath
from omegaconf import DictConfig
from collections import abc

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img, to_rgb, checkpoint
from modules.blocks.res_block import ResidualBlock
from modules.blocks.deform_conv import deform_conv_nd
from modules.blocks.down import DownBlock
from modules.blocks.up import UpBlock
from modules.blocks.mlp import ConvMLP


class SDBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 num_groups: int = 1,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 modulation_type: str = 'none',
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        out_channels = out_channels if out_channels is not None else in_channels

        self.res_block = ResidualBlock(
            in_channels,
            out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            dim=dim,
            num_groups=num_groups,
            use_checkpoint=False,
            use_conv=use_conv,
        )

        if deformable_group_channels is not None:
            deformable_groups = out_channels // deformable_group_channels

        self.dcn = nn.Sequential(
            group_norm(out_channels, num_groups=num_groups),
            get_act(act),
            deform_conv_nd(
                dim,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=deformable_groups,
                offset_scale=offset_scale,
                fix_center=fix_center,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size,
            ),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == b, c, *...
        x = self.res_block(x)
        h = self.dcn(x)

        h = self.drop_path(h) + x

        return h


class SketchDetectionNetwork(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 num_edge_maps: int,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 num_groups: int = 1,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 modulation_type: str = 'softmax',
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 threshold: float = 0.2,
                 loss_config: DictConfig = None,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval

        self.num_edge_maps = num_edge_maps
        self.threshold = threshold

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

        self.dim = dim
        self.hidden_dims = hidden_dims
        self.num_groups = num_groups
        self.act = act

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                conv_nd(
                    dim,
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                self.encoder.append(
                    SDBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        fix_center=fix_center,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )
                in_ch = out_ch
                skip_dims.append(in_ch)

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        dim=2,
                        pool_type=pool_type,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )
                skip_dims.append(in_ch)

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                in_ch = in_ch + skip_dims.pop()
                self.decoder.append(
                    SDBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        fix_center=fix_center,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )
                in_ch = out_ch

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        scale_factor=2,
                        mode=mode,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )

        self.edge_maps = nn.Sequential(
            group_norm(in_ch, num_groups=num_groups),
            get_act(act),
            conv_nd(
                dim,
                in_ch,
                in_ch,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_ch
            ),
            group_norm(in_ch, num_groups=num_groups),
            conv_nd(
                dim,
                in_ch,
                num_edge_maps,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.edge_directions = nn.Sequential(
            group_norm(in_ch, num_groups=num_groups),
            get_act(act),
            conv_nd(
                dim,
                in_ch,
                in_ch,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_ch
            ),
            group_norm(in_ch, num_groups=num_groups),
            conv_nd(
                dim,
                in_ch,
                num_edge_maps,
                kernel_size=1,
            ),
            nn.Hardtanh(-math.pi, math.pi),
        )

        self.line_transform = ConvMLP(
            in_channels=num_edge_maps * 2,
            embed_dim=num_edge_maps * 2 * 4,
            out_channels=1,
            dropout=dropout,
            act=act,
            num_groups=1,
            dim=2,
            use_checkpoint=use_checkpoint
        )

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        for i, block in enumerate(self.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        edge_maps = self.edge_maps(h)

        mask = torch.where(edge_maps >= 0.2, 1.0, 0.0)
        edge_direction = self.edge_directions(h) * mask

        edge = F.sigmoid(self.line_transform(torch.cat([edge_maps, edge_direction], dim=1)))

        return edge

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        net_loss, net_loss_log = self.loss(preds, labels, imgs, split='train')

        if self.global_step % self.log_interval == 0:
            self.log_img(1. - preds, 'edge')
            self.log_img(1. - labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(net_loss_log, prog_bar=True)

        return net_loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        net_loss, net_loss_log = self.loss(preds, labels, imgs, split='val')

        if self.global_step % self.log_interval == 0:
            self.log_img(1. - preds, 'edge')
            self.log_img(1. - labels, 'label')
            self.log_img(imgs, 'img')

        self.log_dict(net_loss_log,  prog_bar=True)

    @torch.no_grad()
    def log_img(self, x: torch.Tensor, split='img'):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/{split}', x[0].float(), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        opts = [opt_net]

        return opts
