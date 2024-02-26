import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import get_pretrained_model, export_layers, normalize_tensor, spatial_average, get_layer_dims


class LPIPS(pl.LightningModule):
    def __init__(self,
                 net_type='swin_v2_t',
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
        self.lins = nn.ModuleList()
        self.rankLoss = BCERankingLoss()

        self.dims = get_layer_dims(net_type)

        for i, dim in enumerate(self.dims):
            self.lins.append(NetLinLayer(dim, dropout=dropout))

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
        acc_r = torch.mean(d1_lt_d0 * judge_per + ~d1_lt_d0 * (1 - judge_per))
        return acc_r

    def forward(self, in0, in1):
        in0, in1 = self.scaling_layer(in0), self.scaling_layer(in1)
        diffs = []

        for feat, lin in zip(self.layers, self.lins):
            in0, in1 = feat(in0), feat(in1)
            diff = torch.abs(normalize_tensor(in0) - normalize_tensor(in1)) ** 2
            diffs.append(spatial_average(lin(diff), keepdim=True))

        val = diffs[0]
        for i in range(1, len(diffs)):
            val += diffs[i]

        return val

    def to(self, device='cuda', *args, **kwargs):
        super().to(device=None, *args, **kwargs)
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

        loss = torch.mean(self.rankLoss(d0, d1, in_judge))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc_avg', self.train_acc_avg / self.train_iter, prog_bar=True, logger=True)

        if self.train_iter % self.log_interval == 0:
            self.train_acc_avg /= self.train_iter
            self.train_iter %= self.log_interval

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

        loss = torch.mean(self.rankLoss(d0, d1, in_judge))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_avg', self.val_acc_avg / self.val_iter, prog_bar=True, logger=True)

        if self.val_iter % self.log_interval == 0:
            self.val_acc_avg /= self.val_iter
            self.val_iter %= self.log_interval

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

        loss = torch.mean(self.rankLoss(d0, d1, in_judge * 2. - 1.))

        self.log(f'{prefix}/loss', loss.detach().mean(), prog_bar=True, logger=True)
        self.log(f'{prefix}/acc_r', acc_r, prog_bar=False, logger=True)
        self.log(f'{prefix}/acc_avg', self.test_acc_avg / self.test_iter, prog_bar=True, logger=True)

        if self.test_iter % self.log_interval == 0:
            self.test_acc_avg /= self.test_iter
            self.test_iter %= self.log_interval

        return self.log_dict

    def on_test_epoch_end(self):
        self.test_iter = 0
        self.test_acc_avg = 0

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.lins.parameters()) +
            list(self.rankLoss.net.parameters()),
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
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self,
                 chn_in,
                 chn_out=1,
                 dropout=0.0,
                 ):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(dropout)]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2, True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2, True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps)),dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self,
                 chn_mid=32,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):
        logit = self.net(d0, d1)
        return self.loss(logit, judge)