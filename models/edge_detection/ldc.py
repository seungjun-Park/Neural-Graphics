import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, Tuple, List, Dict, Any
from utils import conv_nd, get_act, group_norm, pool_nd, conv_transpose_nd
from utils.loss import cats_loss


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


class DSNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int = 2,
                 use_conv: bool = False,
                 pool_type: str = 'max',
                 ):
        super().__init__()

        if use_conv:
            self.pooling = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        else:
            self.pooling = pool_nd(pool_type, dim=dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.pooling(x)

        return x


class USNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_blocks: int = 1,
                 mode: str = 'nearest',
                 act: str = 'relu',
                 dim: int = 2,
                 ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.num_blocks = num_blocks
        self.mode = mode.lower()

        in_ch = in_channels

        self.pads = [0, 0, 1, 3, 7]

        for i in range(num_blocks):
            layer = list()
            pad = self.pads[num_blocks]
            if i == num_blocks - 1:
                out_ch = 1
            else:
                out_ch = 16

            layer.append(conv_nd(dim, in_ch, out_ch, kernel_size=1, stride=1))
            layer.append(get_act(act))
            layer.append(conv_transpose_nd(dim, out_ch, out_ch, kernel_size=2 ** num_blocks, stride=2, padding=pad))

            in_ch = out_ch

            self.layers.append(nn.Sequential(*layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            x = module(x)
        return x


class SingleConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 stride: int = 1,
                 use_norm: bool = True,
                 num_groups: int = None,
                 dim: int = 2,
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.use_norm = use_norm

        num_groups = out_channels if num_groups is None else num_groups

        self.conv = conv_nd(dim, in_channels, out_channels, kernel_size=1, stride=stride)
        if use_norm:
            self.norm = group_norm(out_channels, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)

        return x


class DoubleConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 32,
                 out_channels: int = None,
                 stride: int = 1,
                 num_groups: int = None,
                 use_act: bool = True,
                 act: str = 'relu',
                 dim: int = 2,
                 ):
        super().__init__()

        self.use_act = use_act
        out_channels = embed_dim if out_channels is None else out_channels
        self.conv1 = conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=stride, padding=1)
        self.norm1 = group_norm(embed_dim, embed_dim if num_groups is None else num_groups)
        self.conv2 = conv_nd(dim, embed_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = group_norm(out_channels, out_channels if num_groups is None else num_groups)
        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.use_act:
            x = self.act(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 32,
                 out_channels: int = None,
                 stride: int = 1,
                 num_blocks: int = 2,
                 use_norm: bool = True,
                 num_groups: int = None,
                 use_act: bool = True,
                 act: str = 'relu',
                 dim: int = 2,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_ch = in_channels

        out_channels = embed_dim if out_channels is None else out_channels

        for i in range(num_blocks):
            self.blocks.append(
                DoubleConvBlock(
                    in_channels=in_ch,
                    embed_dim=embed_dim,
                    out_channels=out_channels,
                    stride=stride,
                    num_groups=num_groups,
                    use_act=use_act,
                    act=act,
                    dim=dim
                )
            )

            in_ch = out_channels

        self.skip = SingleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            use_norm=use_norm,
            num_groups=num_groups,
            dim=dim
        )

    def forward(self, x, res):
        residual = self.skip(x)

        for module in self.blocks:
            x = module(x)
            x = 0.5 * (x + residual)

        return x


class LDC(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 96),
                 num_blocks: int = 2,
                 num_downs: int = 2,
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 mode: str = 'nearest',
                 use_norm: bool = True,
                 num_groups: int = None,
                 use_act: bool = True,
                 act: str = 'relu',
                 dim: int = 2,
                 lr: float = 1e-4,
                 weight_decay: float = 0.,
                 log_interval: int = 100,
                 bdr_factor: float = 0.,
                 tex_factor: float = 0.,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.bdr_factor = bdr_factor
        self.tex_factor = tex_factor

        self.blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.us_blocks = nn.ModuleList()
        self.left_skips = nn.ModuleList()
        self.right_skips = nn.ModuleList()
        self.down_skips = nn.ModuleList()

        self.blocks.append(
            DoubleConvBlock(
                in_channels=in_channels,
                embed_dim=embed_dim,
                stride=2,
                num_groups=num_groups,
                use_act=use_act,
                act=act,
                dim=dim
            )
        )

        up_scale = 1
        in_ch = embed_dim

        self.us_blocks.append(
            USNet(
                in_channels=in_ch,
                num_blocks=up_scale,
                mode=mode,
                act=act,
                dim=dim
            )
        )

        for i, out_ch in enumerate(hidden_dims):
            if i < 1:
                self.blocks.append(
                    DoubleConvBlock(
                        in_channels=in_ch,
                        embed_dim=out_ch,
                        num_groups=num_groups,
                        use_act=use_act,
                        act=act,
                        dim=dim,
                    )
                )

            else:
                self.blocks.append(
                    DenseBlock(
                        in_channels=in_ch,
                        embed_dim=out_ch,
                        num_blocks=num_blocks,
                        num_groups=num_groups,
                        use_norm=use_norm,
                        use_act=use_act,
                        act=act,
                        dim=dim,
                    )
                )

                self.right_skips.append(
                    SingleConvBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=1,
                        use_norm=use_norm,
                        num_groups=num_groups
                    )
                )

            if i != len(hidden_dims) - 1:
                self.left_skips.append(
                    SingleConvBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=1,
                        use_norm=use_norm,
                        num_groups=num_groups
                    )
                )

            self.us_blocks.append(
                USNet(
                    in_channels=out_ch,
                    num_blocks=up_scale,
                    mode=mode,
                    act=act,
                    dim=dim
                )
            )

            if i < num_downs:
                if len(self.down_blocks) > 0:
                    self.down_skips.append(
                        SingleConvBlock(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            stride=2,
                            use_norm=use_norm,
                            num_groups=num_groups
                        )
                    )
                self.down_blocks.append(
                    DSNet(
                        in_channels=out_ch,
                        dim=2,
                        use_conv=use_conv,
                        pool_type=pool_type,
                    )
                )
                up_scale += 1

            in_ch = out_ch

        self.co_fusion = CoFusion(
            in_channels=len(hidden_dims) + 1,
            out_channels=1,
            num_groups=num_groups,
            act=act,
            dim=dim,
        )

    def forward(self, x: torch.Tensor):
        edges = []
        for i, (block, us_block) in enumerate(zip(self.blocks, self.us_blocks)):
            if i < 1:
                x = block(x)
                edges.append(us_block(x))
                continue

            if i != len(self.blocks) - 1:
                left_res = self.left_skips[i - 1](x)
                x = block(x)
                edges.append(us_block(x))
                if i < len(self.down_blocks) + 1:
                    x = self.down_blocks[i - 1](x)

                x = x + left_res

            else:
                x = block(x)
                edges.append(us_block(x))
                if i < len(self.down_blocks) + 1:
                    x = self.down_blocks[i - 1](x)

        edges = torch.cat(edges, dim=1)
        edge = self.co_fusion(edges)

        return edge

    def training_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, gt, edge)

        loss = cats_loss(edge, gt, (self.bdr_factor, self.tex_factor))

        self.log('train/loss', loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        self.log_img(img, gt, edge)

        loss = cats_loss(edge, gt, (self.bdr_factor, self.tex_factor))

        self.log('val/loss', loss, logger=True)

        return self.log_dict

    @torch.no_grad()
    def log_img(self, img, gt, edge):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', torch.clamp(img, 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', torch.clamp(gt, 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge', torch.clamp(edge, 0, 1)[0], self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.down_blocks.parameters()) +
                                list(self.blocks.parameters()) +
                                list(self.left_skips.parameters()) +
                                list(self.us_blocks.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )

        return opt
