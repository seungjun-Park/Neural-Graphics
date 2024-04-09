import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, conv_nd, get_act, group_norm
from modules.blocks import ResidualBlock, DownBlock, UpBlock, AttnBlock, WindowAttnBlock, PatchEmbedding, PatchMerging, PatchExpanding
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss, weights_init, LPIPS
from utils.loss import cats_loss, bdcn_loss2


class CoFusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 32,
                 out_channels: int = None,
                 num_groups: int = None,
                 act: str = 'relu',
                 dim: int = 2,
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1)  # before 64
        self.conv2 = conv_nd(dim, embed_dim, out_channels, kernel_size=3, stride=1, padding=1)  # before 64  instead of 32
        self.act = get_act(act)
        num_groups = embed_dim if num_groups is None else num_groups
        self.norm = group_norm(embed_dim, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.act(self.norm(self.conv1(x)))
        attn = F.softmax(self.conv2(attn), dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)


class USBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 16,
                 num_upscale: int = 1,
                 mode: str = 'nearest',
                 dim: int = 2,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.mode = mode.lower()

        in_ch = in_channels

        for i in range(num_upscale):
            out_ch = 1 if i == num_upscale - 1 else embed_dim
            self.blocks.append(conv_nd(dim, in_ch, embed_dim, kernel_size=1, stride=1, padding=0))
            self.blocks.append(conv_nd(dim, embed_dim, out_ch, kernel_size=3, stride=1, padding=1))
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, module in enumerate(self.blocks):
            if i % 2 != 0:
                x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
            x = module(x)
        return F.hardtanh(x, 0, 1)


class EdgeNet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 loss_config: dict = None,
                 out_channels: int = None,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 64,
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
                 lr: float = 2e-5,
                 weight_decay: float = 0.,
                 l_weight: Union[float, List[float], Tuple[float]] = 0.,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 dim: int = 2,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.automatic_optimization = False

        self.l_weight = l_weight

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            in_resolution=in_res,
            embed_dim=embed_dim,
            patch_size=patch_size,
            dim=dim
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = self.patch_embed.patch_res

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

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
                    )
                )
                in_ch = out_ch

                down.append(
                    nn.Sequential(
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                    )
                )

                skip_dims.append(in_ch)
                self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                skip_dims.append(in_ch)
                cur_res = [cur_res[0] // 2, cur_res[1] // 2]

        self.middle = nn.Sequential(
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                groups=groups,
            ),
            WindowAttnBlock(
                in_channels=in_ch,
                in_res=cur_res,
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
            ),
            WindowAttnBlock(
                in_channels=in_ch,
                in_res=cur_res,
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
            ),
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                groups=groups,
            ),
        )

        hidden_dims = hidden_dims[1:]
        hidden_dims.insert(0, embed_dim)

        for i, out_ch in enumerate(reversed(hidden_dims)):
            if i != 0:
                skip_dim = skip_dims.pop()
                self.decoder.append(UpBlock(in_ch + skip_dim, in_ch, dim=dim, mode=mode))
                cur_res = [cur_res[0] * 2, cur_res[1] * 2]

            for j in range(num_blocks):
                up = list()
                skip_dim = skip_dims.pop()
                up.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        groups=groups,
                    )
                )
                in_ch = out_ch

                up.append(
                    nn.Sequential(
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                    )
                )

                self.decoder.append(nn.Sequential(*up))

        up = list()
        for i in range(patch_size // 2):
            skip_dim = skip_dims.pop() if len(skip_dims) > 0 else 0
            up.append(UpBlock(in_ch + skip_dim, in_ch, dim=dim, mode=mode))
            cur_res = [cur_res[0] * 2, cur_res[1] * 2]

            for j in range(num_blocks):
                up.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        dropout=dropout,
                        act=act,
                        groups=groups,
                    )
                )
                up.append(
                    nn.Sequential(
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=cur_res,
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
                        ),
                    )
                )

            up.append(conv_nd(dim, in_ch, out_channels, kernel_size=3, stride=1, padding=1))

        self.decoder.append(nn.Sequential(*up))

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
        x = self.patch_embed(x)
        hs.append(x)

        for i, block in enumerate(self.encoder):
            x = block(x)
            hs.append(x)

        x = self.middle(x)

        for i, block in enumerate(self.decoder):
            x = torch.cat([x, hs.pop()], dim=1)
            x = block(x)

        return x

    def training_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, gt, edge)

        # cats_l = sum([cats_loss(edge, gt, l_weight) for edge, l_weight in zip(edges, self.l_weight)])
        # loss = bdcn_loss2(edge_map, gt, self.l_weight)

        g_loss, g_loss_log = self.loss(gt, edge, cond=img, global_step=self.global_step, optimizer_idx=0, split='train')
        d_loss, d_loss_log = self.loss(gt, edge, cond=img, global_step=self.global_step, optimizer_idx=1, split='train')

        # loss = cats_l + g_loss

        opt_net, opt_disc = self.optimizers()

        opt_net.zero_grad()
        self.manual_backward(g_loss)
        opt_net.step()


        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()

        self.log('train/loss', g_loss, logger=True)
        self.log_dict(g_loss_log)
        self.log_dict(d_loss_log)

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, gt, cond = batch
        edge = self(img)

        self.log_img(img, gt, edge)

        # cats_l = sum([cats_loss(edge, gt, l_weight) for edge, l_weight in zip(edges, self.l_weight)])
        # loss = bdcn_loss2(edge_map, gt, self.l_weight)

        g_loss, g_loss_log = self.loss(gt, edge, cond=img, global_step=self.global_step, optimizer_idx=0,
                                       split='val')
        d_loss, d_loss_log = self.loss(gt, edge, cond=img, global_step=self.global_step, optimizer_idx=1,
                                       split='val')

        # loss = cats_l + g_loss

        self.log('val/loss', g_loss)
        self.log_dict(g_loss_log)
        self.log_dict(d_loss_log)

        return self.log_dict

    @torch.no_grad()
    def log_img(self, img, gt, edges):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', torch.clamp(img, 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', torch.clamp(gt, 0, 1)[0], self.global_step, dataformats='CHW')
        if isinstance(edges, list):
            for i in range(len(edges)):
                tb.add_image(f'{prefix}/side_edge_{i}', torch.clamp(edges[i], 0, 1)[0], self.global_step,
                             dataformats='CHW')
        else:
            tb.add_image(f'{prefix}/edge', torch.clamp(edges, 0, 1)[0], self.global_step, dataformats='CHW')

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def configure_optimizers(self) -> Any:
        opt_net = torch.optim.AdamW(list(self.patch_embed.parameters()) +
                                    list(self.encoder.parameters()) +
                                    list(self.middle.parameters()) +
                                    list((self.decoder.parameters())) +
                                    list(self.out.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    )

        opt_disc = torch.optim.AdamW(list(self.loss.discriminator.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     )

        return [opt_net, opt_disc]
