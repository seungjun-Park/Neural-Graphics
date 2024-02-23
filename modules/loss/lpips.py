import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import get_pretrained_model, export_layers, normalize_tensor, spatial_average, get_layer_dims, BCERankingLoss


class LPIPS(pl.LightningModule):
    def __init__(self,
                 net_type='swin-v2-t',
                 dropout=0.0,
                 log_interval=100,
                 lr=2e-5,
                 weight_decay=0.0,
                 ckpt_path=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.log_interval = log_interval
        self.lr = lr
        self.weight_decay = weight_decay

        net = get_pretrained_model(net_type).eval()
        self.layers = export_layers(net, net_type)

        self.scaling_layer = ScalingLayer()
        self.lin = nn.ModuleList()
        self.rankLoss = BCERankingLoss()

        self.dims = get_layer_dims(net)

        for i, dim in enumerate(self.dims):
            self.lin.append(NetLinLayer(dim, dropout=dropout))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

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
        d1_lt_d0 = (d1 < d0).detach().flatten()
        judge_per = judge.detach().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def forward(self, in0, in1):
        in0, in1 = self.scaling_layer(in0), self.scaling_layer(in1)
        diffs = []

        for feat, lin in zip(self.layers, self.lin):
            feat0, feat1 = module(in0), module(in1)
            diff = (normalize_tensor(feat0.contiguous()) - normalize_tensor(feat1.contiguous())) ** 2
            diffs.append(spatial_average(lin(diff), keepdim=True))

        val = diffs[0]
        for i in range(1, len(diffs)):
            val += diffs[i]

        return val

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        prefix = 'train'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)

        loss = self.rankLoss(d0, d1, in_judge * 2. - 1.)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        prefix = 'val'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)

        loss = torch.mean(self.rankLoss(d0, d1, in_judge * 2. - 1.))

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = dict()
        prefix = 'test'
        in_ref, in_p0, in_p1, in_judge = batch

        d0 = self(in_ref, in_p0)
        d1 = self(in_ref, in_p1)
        acc_r = self.compute_accuracy(d0, d1, judge=in_judge)

        loss = torch.mean(self.rankLoss(d0, d1, in_judge * 2. - 1.))

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            lr=self.lr,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay
        )

        return [opt], []


class ScalingLayer(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(ScalingLayer, self).__init__(*args, **kwargs)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None])
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None])

    def forward(self, inp):
        inp = (inp - self.mean) / self.std
        inp = F.interpolate(inp, size=(224, 224), antialias=True, mode='bicubic', align_corners=True)
        return inp


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self,
                 chn_in,
                 chn_out=1,
                 dropout=0.0,
                 ):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(dropout)]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)