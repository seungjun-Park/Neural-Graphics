import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import get_pretrained_model, export_layers, normalize_tensor, spatial_average, get_layer_dims
from modules.blocks import AttnBlock


class LPIPS(pl.LightningModule):
    def __init__(self,
                 loss_config=None,
                 net_type: str = 'swin_v2_t',
                 dropout: float = 0.5,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 log_interval: int = 100,
                 lr: float = 2e-5,
                 weight_decay: float = 0.0,
                 ckpt_path: str = None,
                 ignore_keys=[],
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.log_interval = log_interval
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_iter = 0
        self.train_acc_avg = 0
        self.val_iter = 0
        self.val_acc_avg = 0
        self.test_iter = 0
        self.test_acc_avg = 0

        net_type = net_type.lower()
        net = get_pretrained_model(net_type).eval()
        self.layers = export_layers(net, net_type)

        self.scaling_layer = ScalingLayer()
        self.attns = nn.ModuleList()

        self.dims = get_layer_dims(net_type)

        for i, dim in enumerate(self.dims):
            self.attns.append(
                AttentionLayer(
                    dim,
                    out_channels=1,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    bias=bias
                )
            )

        if loss_config is not None:
            self.loss = BCERankingLoss(**loss_config)
        else:
            self.loss = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    @torch.no_grad()
    def compute_accuracy(self, d0, d1, judge):
        d0 = d0.cpu().detach().flatten()
        d1 = d1.cpu().detach().flatten()
        judge_per = judge.cpu().detach().flatten()
        acc_r = torch.mean((d1 < d0) * judge_per + (d1 > d0) * (1 - judge_per)) #+ (d1 == d0) * 0.5)
        return acc_r

    def forward(self, in0, in1):
        in0, in1 = self.scaling_layer(in0), self.scaling_layer(in1)
        diffs = []

        for feat, attn in zip(self.layers, self.attns):
            in0, in1 = feat(in0), feat(in1)
            diff = torch.abs(in0 - in1)
            diffs.append(spatial_average(attn(diff), keepdim=True))

        val = diffs[0]
        for i in range(1, len(diffs)):
            val += diffs[i]

        return val

    def to(self, device, *args, **kwargs):
        super().to(device=device, *args, **kwargs)
        for module in self.layers:
            module.to(device=device)

    def on_train_start(self):
        for module in self.layers:
            module.to(device=self.device)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        prefix = 'train'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)
        self.train_acc_avg += acc_r
        self.train_iter += 1

        loss = torch.mean(self.loss(d0, d1, in_judge))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc_avg', self.train_acc_avg / self.train_iter, prog_bar=True, logger=True)

        if self.train_iter % self.log_interval == 0:
            self.train_acc_avg /= self.train_iter
            self.train_iter = 1

        return loss

    def on_train_epoch_end(self):
        self.train_iter = 0
        self.train_acc_avg = 0

    def on_validation_start(self):
        for module in self.layers:
            module.to(device=self.device)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        prefix = 'val'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)
        self.val_acc_avg += acc_r
        self.val_iter += 1

        loss = torch.mean(self.loss(d0, d1, in_judge))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_avg', self.val_acc_avg / self.val_iter, prog_bar=True, logger=True)

        if self.val_iter % self.log_interval == 0:
            self.val_acc_avg /= self.val_iter
            self.val_iter = 1

        return self.log_dict

    def on_validation_epoch_end(self):
        self.val_iter = 0
        self.val_acc_avg = 0

    def on_test_start(self):
        for module in self.layers:
            module.to(device=self.device)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        prefix = 'test'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)
        self.test_acc_avg += acc_r
        self.test_iter += 1

        loss = torch.mean(self.loss(d0, d1, in_judge * 2. - 1.))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc_avg', self.test_acc_avg / self.test_iter, prog_bar=True, logger=True)

        if self.test_iter % self.log_interval == 0:
            self.test_acc_avg /= self.test_iter
            self.test_iter = 1

        return self.log_dict

    def on_test_epoch_end(self):
        self.test_iter = 0
        self.test_acc_avg = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            list(self.attns.parameters()) +
            list(self.loss.net.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay
        )

        return [opt], []


class ScalingLayer(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(ScalingLayer, self).__init__(*args, **kwargs)
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single layer which does a multi heads self attention. '''
    def __init__(self,
                 chn_in,
                 chn_out=1,
                 dropout=0.5,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        layers = [nn.Dropout(dropout), ]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AttentionLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0,
                 attn_dropout: float = 0,
                 bias: bool = True
                 ):
        super().__init__()

        out_channels = out_channels if out_channels is not None else in_channels

        self.attn = nn.MultiheadAttention(in_channels, 1, dropout=attn_dropout, bias=bias, batch_first=True)
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(b, -1, *spatial)

        return x


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self,
                 embed_dim: int = 32,
                 num_block: int = 2,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 *args,
                 **kwargs,
                 ):
        super(Dist2LogitLayer, self).__init__(*args, **kwargs)

        layers = [nn.Conv2d(5, embed_dim, 1, bias=bias)]
        for i in range(num_block):
            layers += [
                AttentionLayer(
                    embed_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    bias=bias
                )
            ]

        layers += [nn.Conv2d(embed_dim, 1, 1, bias=bias)]
        layers += [nn.Softmax(dim=1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=1e-5):
        inp = torch.cat((d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps)), dim=1)
        inp = self.layers(inp)

        return inp


class BCERankingLoss(nn.Module):
    def __init__(self,
                 embed_dim=32,
                 num_block: int = 2,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 bias: bool = True,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.net = Dist2LogitLayer(
            embed_dim=embed_dim,
            num_block=num_block,
            dropout=dropout,
            attn_dropout=attn_dropout,
            bias=bias,
        )
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):
        logit = self.net(d0, d1)
        return self.loss(logit, judge)