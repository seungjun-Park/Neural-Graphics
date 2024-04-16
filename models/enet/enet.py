import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm, normalize_img
from modules.blocks import (
    ResidualBlock, DownBlock, UpBlock, AttnBlock,
    WindowAttnBlock, PatchEmbedding, PatchMerging, PatchExpanding,
    ScaledSkipBlock
)


class EdgeNet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 loss_config: dict = None,
                 out_channels: int = None,
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
                 groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 mode: str = 'nearest',
                 lr: Union[float, List[float], Tuple[float]] = 2e-5,
                 weight_decay: Union[float, List[float], Tuple[float]] = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 use_fp16: bool = False,
                 accum_grad_batches: int = 1,
                 ):
        super().__init__()

        self.automatic_optimization = False

        self.accum_grad_batches = accum_grad_batches

        self.lr = to_2tuple(lr)
        self.weight_decay = to_2tuple(weight_decay)
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.num_blocks = num_blocks
        self.dim = dim
        self.mode = mode
        self.pool_type = pool_type
        self.use_conv = use_conv
        self.act = act
        self.groups = groups

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        self.embed = conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = in_res

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.scaled_skip_blocks = nn.ModuleList()

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                down = list()
                down.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        groups=groups,
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
                                shift_size=0,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_conv=False,
                                dim=dim,
                                use_checkpoint=use_checkpoint,
                            ),
                            WindowAttnBlock(
                                in_channels=in_ch,
                                in_res=to_2tuple(cur_res),
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                window_size=window_size,
                                shift_size=window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_conv=False,
                                dim=dim,
                                use_checkpoint=use_checkpoint,
                            ),
                        )
                    )

                skip_dims.append(in_ch)
                self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                skip_dims.append(in_ch)
                cur_res //= 2

        skip_dim = sum([i for i in skip_dims])

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            self.scaled_skip_blocks.append(
                ScaledSkipBlock(
                    level=i,
                    skip_dims=skip_dims,
                    num_blocks=num_blocks,
                    act=act,
                    num_groups=groups,
                    dim=dim,
                    use_conv=use_conv,
                    pool_type=pool_type,
                    mode=mode,
                    use_checkpoint=use_checkpoint
                )
            )

            for j in range(num_blocks):
                up = list()
                up.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        groups=groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                    )
                )
                in_ch = out_ch

                if i in attn_res:
                    up.append(
                        nn.Sequential(
                            WindowAttnBlock(
                                in_channels=in_ch,
                                in_res=to_2tuple(cur_res),
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                window_size=window_size,
                                shift_size=0,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_conv=False,
                                dim=dim,
                                use_checkpoint=use_checkpoint,
                            ),
                            WindowAttnBlock(
                                in_channels=in_ch,
                                in_res=to_2tuple(cur_res),
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                window_size=window_size,
                                shift_size=window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=bias,
                                drop=dropout,
                                attn_drop=attn_dropout,
                                drop_path=drop_path,
                                act=act,
                                use_conv=False,
                                dim=dim,
                                use_checkpoint=use_checkpoint,
                            ),
                        )
                    )

                self.decoder.append(nn.Sequential(*up))

            if i != 0:
                self.decoder.append(UpBlock(in_ch + skip_dim, out_channels=in_ch, dim=dim, mode=mode, use_conv=use_conv))
                cur_res = int(cur_res * 2)

        self.out = nn.Sequential(
            group_norm(in_ch, groups),
            get_act(act),
            conv_nd(dim, in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        )

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
        hs = []
        h = self.embed(x)
        hs.append(h)
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        h_cats = []
        for i, block in enumerate(self.scaled_skip_blocks):
            h_cats.append(block(hs))

        for i, block in enumerate(self.decoder):
            h = torch.cat([h, ] + h_cats[i // (self.num_blocks + 1)], dim=1)
            h = block(h)

        h = self.out(h)

        return F.sigmoid(h)

    def training_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, gt, edge)

        opt_net, opt_disc = self.optimizers()

        loss, loss_log = self.loss(gt, edge, img, self.global_step, 0, split='train', last_layer=self.get_last_layer())
        loss = loss / self.accum_grad_batches

        d_loss, d_loss_log = self.loss(gt, edge, img, self.global_step, 1, split='train', last_layer=self.get_last_layer())
        d_loss = d_loss / self.accum_grad_batches

        self.manual_backward(loss)
        self.manual_backward(d_loss)

        if (batch_idx + 1) % self.accum_grad_batches == 0:
            opt_net.step()
            opt_net.zero_grad()
            opt_disc.step()
            opt_disc.zero_grad()

        self.log_dict(d_loss_log)
        self.log_dict(loss_log)

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, gt, cond = batch
        edge = self(img)
        self.log_img(img, gt, edge)

        loss, loss_log = self.loss(gt, edge, img, self.global_step, 0, split='train', last_layer=self.get_last_layer())
        d_loss, d_loss_log = self.loss(gt, edge, img, self.global_step, 1, split='train',
                                       last_layer=self.get_last_layer())
        self.log_dict(loss_log)
        self.log_dict(d_loss_log)

        return self.log_dict

    @torch.no_grad()
    def log_img(self, img, gt, edges):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', img[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', gt[0], self.global_step, dataformats='CHW')
        if isinstance(edges, list):
            for i in range(len(edges)):
                tb.add_image(f'{prefix}/side_edge_{i}', edges[i][0, ...], self.global_step,
                             dataformats='CHW')
        else:
            tb.add_image(f'{prefix}/edge', edges[0], self.global_step, dataformats='CHW')

    def get_last_layer(self):
        return self.out[-1].weight

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.embed.parameters()) +
                                    list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.scaled_skip_blocks.parameters()) +
                                    list(self.out.parameters()),
                                    lr=self.lr[0],
                                    weight_decay=self.weight_decay[0],
                                    betas=(0.5, 0.9)
                                    )

        opt_disc = torch.optim.AdamW(list(self.loss.disc.parameters()),
                                     lr=self.lr[1],
                                     weight_decay=self.weight_decay[1],
                                     betas=(0.5, 0.9)
                                     )

        # lr_net = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=opt_net,
        #     lr_lambda=lambda epoch: 1.0 if epoch < self.lr_decay_epoch else (0.95 ** (epoch - self.lr_decay_epoch))
        # )

        return [opt_net, opt_disc]#, [{"scheduler": lr_net, "interval": "epoch"}]
