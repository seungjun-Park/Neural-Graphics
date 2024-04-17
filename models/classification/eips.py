import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img
from modules.blocks import (
    ResidualBlock, DownBlock,
    WindowAttnBlock,
)


class EIPS(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 pretrained_window_size: Union[int, List[int], Tuple[int]] = 0,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 16,
                 attn_res: Union[List[int], Tuple[int]] = (2, 3),
                 num_blocks: int = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 attn_mode: str = 'cosine',  # 'cosine' is swin-v2 attn block, 'vanilla' is swin-v1 attn block
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.num_blocks = num_blocks
        self.dim = dim
        self.pool_type = pool_type
        self.use_conv = use_conv
        self.act = act
        self.num_groups = num_groups

        self.in_channels = in_channels
        self.in_res = in_res
        self.hidden_dims = hidden_dims

        self.embed = conv_nd(dim, in_channels, embed_dim, kernel_size=1, stride=1)

        in_ch = embed_dim
        cur_res = in_res

        self.encoder = nn.ModuleList()

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                down = list()
                down.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        use_conv=use_conv,
                    )
                )
                in_ch = out_ch

                if i in attn_res:
                    down.append(
                        nn.Sequential(
                            WindowAttnBlock(
                                in_channels=in_ch,
                                in_res=to_2tuple(cur_res),
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                window_size=window_size,
                                pretrained_windows_size=pretrained_window_size,
                                shift_size=0,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_checkpoint=use_checkpoint,
                                attn_mode=attn_mode,
                            ),
                            WindowAttnBlock(
                                in_channels=in_ch,
                                in_res=to_2tuple(cur_res),
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                window_size=window_size,
                                pretrained_windows_size=pretrained_window_size,
                                shift_size=window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_checkpoint=use_checkpoint,
                                attn_mode=attn_mode,
                            ),
                        )
                    )

                self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                cur_res //= 2

        self.norm = group_norm(in_ch, num_groups=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(in_ch, 1)

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

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> List[torch.Tensor]:
        h = self.embed(x.type(self.dtype))
        for i, block in enumerate(self.encoder):
            h = block(h)

        b, c, h, w = h.shape
        h = h.reshape(b, c, -1)
        h = self.norm(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)

        return self.out(h)

    def training_step(self, batch, batch_idx):
        img, edge, label = batch

        logit = self(torch.cat([img, edge], dim=1))
        loss = F.binary_cross_entropy_with_logits(logit, label)

        self.log('train/loss', loss, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, edge, label = batch

        logit = self(torch.cat([img, edge], dim=1))
        loss = F.binary_cross_entropy_with_logits(logit, label)  # To stable training, when fp16 used.

        self.log('train/val', loss, logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.embed.parameters()) +
                                    list(self.encoder.parameters()) +
                                    list(self.out.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9)
                                    )

        # lr_net = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=opt_net,
        #     lr_lambda=lambda epoch: 1.0 if epoch < self.lr_decay_epoch else (0.95 ** (epoch - self.lr_decay_epoch))
        # )

        return [opt_net]#, [{"scheduler": lr_net, "interval": "epoch"}]