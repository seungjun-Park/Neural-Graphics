import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, Tuple, List, Dict, Any
from utils import conv_nd, get_act, group_norm, pool_nd


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed

    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost

# ------------ cats losses ----------


def bdrloss(prediction, label, radius):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1).to(prediction.device)
    filt.requires_grad = False

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return torch.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3).to(prediction.device)
    filt1.requires_grad = False

    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1).to(prediction.device)
    filt2.requires_grad = False

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))


def cats_loss(prediction, label, l_weight=(0., 0.)):
    # tracingLoss

    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    prediction = torch.sigmoid(prediction)

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4)

    return cost + bdr_factor * bdrcost + tex_factor * textcost


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

        self.use_conv = use_conv

        if use_conv:
            self.pooling = conv_nd(dim, in_channels, in_channels, kernel_size=3, stride=2, padding=0)

        else:
            self.pooling = pool_nd(pool_type, dim=dim)

    def forward(self, x):
        if self.use_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.pooling(x)

        else:
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

        for i in range(num_blocks):
            layer = list()
            if i == num_blocks - 1:
                out_ch = 1
            else:
                out_ch = in_ch // 2

            layer.append(conv_nd(dim, in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            layer.append(get_act(act))

            in_ch = out_ch

            self.layers.append(nn.Sequential(*layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
            x = module(x)
        x = F.hardtanh(x, 0, 1)
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

    def forward(self, x):
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
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval

        self.blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.us_blocks = nn.ModuleList()

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

            in_ch = out_ch

            self.us_blocks.append(
                USNet(
                    in_channels=in_ch,
                    num_blocks=up_scale,
                    mode=mode,
                    act=act,
                    dim=dim
                )
            )

            if i < num_downs:
                self.down_blocks.append(
                    DSNet(
                        in_channels=in_ch,
                        dim=2,
                        use_conv=use_conv,
                        pool_type=pool_type,
                    )
                )
                up_scale += 1

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
            x = block(x)
            edges.append(us_block(x))
            if 0 < i < len(self.down_blocks) + 1:
                x = self.down_blocks[i - 1](x)

        edges = torch.cat(edges, dim=1)
        edge = self.co_fusion(edges)

        return edge

    def training_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        if self.global_step % self.log_interval == 0:
            self.log_img(img, gt, cond)

        loss = cats_loss(edge, gt, (1., 1.))

        self.log('train/loss', loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge = self(img)

        self.log_img(img, gt, cond)

        loss = cats_loss(edge, gt, (1., 1.))

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
                                list(self.us_blocks.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )

        return opt