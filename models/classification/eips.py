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

        self.residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            dim=dim,
            num_groups=num_groups,
            use_conv=use_conv,
            use_checkpoint=use_checkpoint,
        )

        self.attn = DoubleWindowSelfAttentionBlock(
            in_channels=out_channels,
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.residual_block(x)
        z, attn_map = self.attn(h)
        z = h + self.drop_path(z)
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
                 pool_type: str = 'conv',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 margin: float = 0.0,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.margin = margin

        assert num_head_channels != -1 or num_heads != -1

        if num_head_channels != -1:
            use_num_head_channels = True
        else:
            use_num_head_channels = False

        self.encoder = nn.ModuleList()
        self.similarity_net = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
                group_norm(embed_dim, num_groups=num_groups),
                get_act(act),
            )
        )

        in_ch = embed_dim
        cur_res = in_res

        cat_dim = 0

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                self.encoder.append(
                    EncoderBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
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
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )

                in_ch = out_ch

                self.similarity_net.append(
                    EncoderBlock(
                        in_channels=in_ch + cat_dim,
                        out_channels=in_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=in_ch // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )

                cat_dim = in_ch

            self.cross_attn_blocks.append(
                DoubleWindowCrossAttentionBlock(
                    in_channels=in_ch,
                    in_res=to_2tuple(cur_res),
                    num_heads=in_ch // num_head_channels if use_num_head_channels else num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    proj_bias=bias,
                    dropout=attn_dropout,
                    use_checkpoint=use_checkpoint,
                    attn_mode=attn_mode,
                    dim=dim
                )
            )

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(in_ch, scale_factor=2.0, dim=dim, num_groups=num_groups, pool_type=pool_type)
                )
                self.similarity_net.append(
                    DownBlock(in_ch, scale_factor=2.0, dim=dim, num_groups=num_groups, pool_type=pool_type)
                )
                cur_res //= 2

        in_ch = int(in_ch * (cur_res ** 2))

        self.out = nn.Linear(in_ch, 1)

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
        feat_imgs = []
        feat_edges = []
        for i, module in enumerate(self.encoder):
            img = module(img)
            edge = module(edge)
            if not isinstance(module, DownBlock):
                feat_imgs.append(img)
                feat_edges.append(edge)

        cross_attns = []
        for i, module in enumerate(self.cross_attn_blocks[::-1]):
            cross_attns.append(module(feat_edges.pop(), feat_imgs.pop())[0])

        x = cross_attns.pop()

        for i, module in enumerate(self.similarity_net):
            if i != 0:
                if isinstance(module, DownBlock):
                    x = module(x)
                else:
                    x = module(torch.cat([x, cross_attns.pop()], dim=1))
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

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.encoder.parameters()) +
                                list(self.similarity_net.parameters()) +
                                list(self.cross_attn_blocks.parameters()) +
                                list(self.out.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )

        return [opt]
