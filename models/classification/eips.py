import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models.vgg import VGG
from omegaconf import DictConfig, ListConfig
from collections import namedtuple
from timm.models.layers import DropPath

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.blocks.res_block import ResidualBlock
from modules.blocks.attn_block import DoubleWindowCrossAttentionBlock, DoubleWindowSelfAttentionBlock
from modules.blocks.down import DownBlock
from modules.blocks.mlp import ConvMLP, MLP
from modules.blocks.patches import PatchMerging


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 mlp_ratio: float = 4.0,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.attn = DoubleWindowSelfAttentionBlock(
            in_channels=in_channels,
            in_res=to_2tuple(in_res),
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            proj_bias=bias,
            dropout=attn_dropout,
            use_checkpoint=use_checkpoint,
            attn_mode=attn_mode,
            dim=dim
        )

        self.mlp = ConvMLP(
            in_channels=in_channels,
            embed_dim=int(in_channels * mlp_ratio),
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            num_groups=num_groups,
            dim=dim,
            use_checkpoint=use_checkpoint
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, attn_map = self.attn(x)
        h = x + self.drop_path(h)
        z = h + self.drop_path(self.mlp(h))

        return z


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 mlp_ratio: float = 4.0,
                 out_channels: int = None,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.attn = DoubleWindowSelfAttentionBlock(
            in_channels=in_channels,
            in_res=to_2tuple(in_res),
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            proj_bias=bias,
            dropout=attn_dropout,
            use_checkpoint=use_checkpoint,
            attn_mode=attn_mode,
            dim=dim
        )

        self.cross_attn = DoubleWindowCrossAttentionBlock(
            in_channels=in_channels,
            in_res=to_2tuple(in_res),
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            proj_bias=bias,
            dropout=attn_dropout,
            use_checkpoint=use_checkpoint,
            attn_mode=attn_mode,
            dim=dim
        )

        self.mlp = ConvMLP(
            in_channels=in_channels,
            embed_dim=int(in_channels * mlp_ratio),
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            num_groups=num_groups,
            dim=dim,
            use_checkpoint=use_checkpoint
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h, attn_map = self.attn(x)
        h = x + self.drop_path(h)
        z, attn_map = self.cross_attn(h, context)
        z = h + self.drop_path(z)
        z = z + self.drop_path(self.mlp(z))
        return z


class EIPS(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 mlp_ratio: float = 4.0,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: int = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval

        assert num_head_channels != -1 or num_heads != -1

        if num_head_channels != -1:
            use_num_head_channels = True
        else:
            use_num_head_channels = False

        self.decoder = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                conv_nd(dim, in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
                group_norm(embed_dim, num_groups=num_groups),
            )
        )

        self.decoder.append(
            nn.Sequential(
                conv_nd(dim, in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
                group_norm(embed_dim, num_groups=num_groups),
            )
        )

        in_ch = embed_dim
        cur_res = in_res // patch_size
        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                self.encoder.append(
                    EncoderBlock(
                        in_channels=in_ch,
                        mlp_ratio=mlp_ratio,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=out_ch // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )

                self.decoder.append(
                    DecoderBlock(
                        in_channels=in_ch,
                        mlp_ratio=mlp_ratio,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=out_ch // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    PatchMerging(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=2,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        dim=dim
                    )
                )
                self.decoder.append(
                    PatchMerging(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=2,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        dim=dim
                    )
                )
                in_ch = out_ch
                cur_res //= 2

        self.out = nn.Linear(int(in_ch * cur_res ** 2), 1)

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

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
        context = img
        x = edge
        contexts = []
        for i, module in enumerate(self.encoder):
            context = module(context)
            if isinstance(module, EncoderBlock):
                contexts.append(context)

        for i, module in enumerate(self.decoder):
            if isinstance(module, DecoderBlock):
                x = module(x, contexts.pop(0))
            else:
                x = module(x)

        x = torch.flatten(x, start_dim=1)
        x = self.out(x)

        return F.sigmoid(x)

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
        opt = torch.optim.AdamW(list(self.decoder.parameters()) +
                                list(self.out.parameters()) +
                                list(self.encoder.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )

        return [opt]
