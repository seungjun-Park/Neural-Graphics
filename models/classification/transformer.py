import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Union, List, Tuple, Dict

from modules.blocks import DoubleWindowAttentionBlock, DownBlock
from utils import instantiate_from_config, to_2tuple, conv_nd, group_norm
from omegaconf import ListConfig


class SwinTransformer(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 16,
                 quant_dim: int = 4,
                 logit_dim: int = 1,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 num_groups: int = 16,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_norm: bool = True,
                 lr: float = 1e-4,
                 weight_decay: float = 0.,
                 ckpt_path: str = None,
                 ignored_keys: list = ['attn_mask']
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.embed = nn.Sequential(
            conv_nd(dim, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            group_norm(embed_dim, num_groups=num_groups)
        )
        self.encoder = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        if isinstance(num_blocks, ListConfig):
            num_blocks = list(num_blocks)

        if isinstance(num_heads, ListConfig):
            num_heads = list(num_heads)

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            for j in range(num_blocks[i] if isinstance(num_blocks, list) else num_blocks):
                down.append(
                    DoubleWindowAttentionBlock(
                        in_channels=in_ch,
                        in_res=to_2tuple(cur_res),
                        out_channels=out_ch,
                        num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads,
                        window_size=window_size,
                        qkv_bias=qkv_bias,
                        proj_bias=bias,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        act=act,
                        num_groups=num_groups,
                        use_norm=use_norm,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                        use_conv=use_conv,
                        dim=dim
                    )
                )
                in_ch = out_ch

            self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, pool_type=pool_type))
                cur_res //= 2

            if use_conv:
                self.quant = conv_nd(dim, in_ch, quant_dim, kernel_size=1, stride=1)
            else:
                self.quant = nn.Linear(in_ch, quant_dim)

            self.logit_out = nn.Linear(int(quant_dim * (cur_res ** 2)), logit_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignored_keys)

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

    def train(self, mode: bool = True):
        super().train(mode)
        for param in self.parameters():
            param.requires_grad = True

        return self

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)

        for i, module in enumerate(self.encoder):
            h = module(h)

        h = self.quant(h)
        h = torch.flatten(h, start_dim=1)
        h = self.logit_out(h)

        return h

    @torch.no_grad()
    def feature_extract(self, x: torch.Tensor, use_deep_supervision: bool = False)\
            -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]:
        h = self.embed(x)
        hs = []
        for i, module in enumerate(self.encoder):
            h = module(h)
            if not isinstance(module, DownBlock):
                hs.append(h)

        if use_deep_supervision:
            return hs
        return h

    def training_step(self, batch, batch_idx):
        x, label = batch
        prob = self(x)

        loss = F.binary_cross_entropy_with_logits(prob, label)

        self.log('train/loss', loss, rank_zero_only=True, logger=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, label = batch
        prob = self(x)

        loss = F.binary_cross_entropy_with_logits(prob, label)

        self.log('val/loss', loss, rank_zero_only=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.99)
                                )

        return opt

