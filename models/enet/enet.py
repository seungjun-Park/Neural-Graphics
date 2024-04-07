import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple
from modules.blocks import ResidualBlock, LearnableFourierMask, DownBlock, UpBlock, AttnBlock
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss, weights_init, LPIPS


class EdgeNet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 disc_config: dict = None,
                 out_channels: int = None,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 64,
                 num_blocks: int = 2,
                 fourier_mask_res: Union[List[int], Tuple[int]] = (),
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 mode: str = 'nearest',
                 lr: float = 2e-5,
                 weight_decay: float = 0.,
                 perceptual_weight: float = 1.0,
                 disc_weight: float = 0.,
                 disc_factor: float = 1.,
                 disc_iter_start: int = 50001,
                 disc_loss: str = 'hinge',
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

        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = LPIPS().eval()

        self.disc = instantiate_from_config(disc_config).apply(weights_init)
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.disc_iter_start = disc_iter_start
        self.disc_loss = hinge_d_loss if disc_loss.lower() == "hinge" else vanilla_d_loss

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.embed_dim = embed_dim

        self.down = nn.ModuleList()
        self.down.append(nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1))

        self.middle = nn.ModuleList()
        self.up = nn.ModuleList()

        skip_dims = [embed_dim]

        in_ch = embed_dim

        current_res = in_res

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

                if i in fourier_mask_res:
                    down.append(
                        LearnableFourierMask(
                            in_channels=in_ch,
                            in_res=current_res
                        )
                    )

                skip_dims.append(in_ch)
                self.down.append(*down)

            self.down.append(DownBlock(in_ch))
            current_res = current_res // 2
            skip_dims.append(in_ch)

        self.middle = nn.Sequential(
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                groups=groups,
            ),
            # LearnableFourierMask(
            #     in_ch,
            #     in_res=current_res
            # ),
            AttnBlock(
                in_ch,
                mlp_ratio=mlp_ratio,
                heads=num_heads,
                num_head_channels=num_head_channels,
                dropout=dropout,
                attn_dropout=attn_dropout,
                bias=bias,
                act=act,
                use_conv=use_conv,
                dim=dim,
                groups=groups,
            ),
            ResidualBlock(
                in_channels=in_ch,
                out_channels=in_ch,
                dropout=dropout,
                act=act,
                groups=groups,
            )
        )

        hidden_dims = hidden_dims[1:]
        hidden_dims.reverse()
        hidden_dims.append(embed_dim)

        for i, out_ch in enumerate(hidden_dims):
            self.up.append(UpBlock(in_ch + skip_dims.pop(), out_channels=in_ch, mode=mode))
            current_res = current_res ** 2

            for j in range(num_blocks):
                up = list()
                up.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        groups=groups,
                    )
                )
                in_ch = out_ch

                if i in fourier_mask_res:
                    up.append(
                        LearnableFourierMask(
                            in_channels=in_ch,
                            in_res=current_res
                        )
                    )
                self.up.append(*up)

        self.out = nn.Conv2d(
            in_ch + skip_dims.pop(),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        hs = []
        for module in self.down:
            x = module(x)
            hs.append(x)

        x = self.middle(x)

        for module in self.up:
            x = torch.cat([x, hs.pop()], dim=1)
            x = module(x)

        x = torch.cat([x, hs.pop()], dim=1)
        x = self.out(x)

        return x

    def training_step(self, batch, batch_idx):
        img, gt, cond = batch
        edge_map = self(img)

        if (self.global_step // 2) % self.log_interval == 0:
            self.log_img(img, gt, edge_map)

        rec_loss = torch.abs(gt.contiguous() - edge_map.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(gt.repeat(1, 3, 1, 1).contiguous(), edge_map.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # generator update
        g_loss = -torch.mean(self.disc(edge_map.contiguous(), img))

        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        disc_factor = self.adopt_weight(self.disc_factor, self.global_step // 2, threshold=self.disc_iter_start)

        loss = d_weight * disc_factor * g_loss + rec_loss

        opt_unet, opt_disc = self.optimizers()

        # train encoder+decoder+logvar
        opt_unet.zero_grad()
        self.manual_backward(loss)
        opt_unet.step()

        # second pass for discriminator update
        logits_real = self.disc(gt.contiguous().detach(), img)
        logits_fake = self.disc(edge_map.contiguous().detach(), img)

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()

        self.log('train/loss', loss, logger=True)

    def validation_step(self, batch, batch_idx) -> Optional[Any]:
        img, gt, cond = batch
        edge_map = self(img)

        if (self.global_step // 2) % self.log_interval == 0:
            self.log_img(img, gt, edge_map)

        rec_loss = torch.abs(gt.contiguous() - edge_map.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(gt.repeat(1, 3, 1, 1).contiguous(), edge_map.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # generator update
        g_loss = -torch.mean(self.disc(edge_map.contiguous(), img))

        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        disc_factor = self.adopt_weight(self.disc_factor, self.global_step // 2, threshold=self.disc_iter_start)

        loss = d_weight * disc_factor * g_loss + rec_loss

        # discriminator update
        logits_real = self.disc(gt.contiguous().detach(), img)
        logits_fake = self.disc(edge_map.contiguous().detach(), img)

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        self.log('val/loss', loss, logger=True)

        return self.log_dict

    def test_step(self, batch, batch_idx) -> Optional[Any]:
        img, gt, cond = batch
        edge_map = self(img)

        if (self.global_step // 2) % self.log_interval == 0:
            self.log_img(img, gt, edge_map)

        rec_loss = torch.abs(gt.contiguous() - edge_map.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(gt.repeat(1, 3, 1, 1).contiguous(), edge_map.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # generator update
        g_loss = -torch.mean(self.disc(edge_map.contiguous(), img))

        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        disc_factor = self.adopt_weight(self.disc_factor, self.global_step // 2, threshold=self.disc_iter_start)

        loss = d_weight * disc_factor * g_loss + rec_loss

        # discriminator update
        logits_real = self.disc(gt.contiguous().detach(), img)
        logits_fake = self.disc(edge_map.contiguous().detach(), img)

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        self.log('test/loss', loss, logger=True)

        return self.log_dict

    def log_img(self, img, gt, edge):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', torch.clamp(img, 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/gt', torch.clamp(gt, 0, 1)[0], self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge', torch.clamp(edge, 0, 1)[0], self.global_step, dataformats='CHW')

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def calculate_adaptive_weight(self, rec_loss, g_loss):
        rec_grads = torch.autograd.grad(rec_loss, self.out.weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.out.weight, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def configure_optimizers(self) -> Any:
        opt_unet = torch.optim.AdamW(list(self.down.parameters()) +
                                     list(self.middle.parameters()) +
                                     list((self.up.parameters())) +
                                     list(self.out.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     )

        opt_disc = torch.optim.AdamW(list(self.disc.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     )

        return [opt_unet, opt_disc]
