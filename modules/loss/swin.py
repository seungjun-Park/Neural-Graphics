import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights


def get_pretrained_model(type):
    if type == 'swin-t':
        return swin_t(weights=Swin_T_Weights.DEFAULT)
    elif type == 'swin-s':
        return swin_s(weights=Swin_S_Weights.DEFAULT)
    elif type == 'swin-b':
        return swin_b(weights=Swin_B_Weights.DEFAULT)
    elif type == 'swin-v2-t':
        return swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    elif type == 'swin-s':
        return swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
    elif type == 'swin-b':
        return swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    else:
        NotImplementedError(f'{self.type} is not available.')



class SwinLoss(nn.Module):
    def __init__(self,
                 type='swin-t',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.type = type.lower()

        swin = get_pretrained_model(self.type).eval()
        features = swin.features.eval()

        self.layers = [
            features[: 2].eval(),
            features[2: 4].eval(),
            features[4: 6].eval(),
            features[6: ].eval(),
        ]

        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def preprocessing(self, x):
        x = (x - self.mean) / self.std
        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False, antialias=True)

        return x

    def forward(self, target, pred):
        b, c, h, w = target.shape

        target = self.preprocessing(target)
        pred = self.preprocessing(pred)

        diffs = []
        for i, module in enumerate(self.layers):
            target = module(target)
            pred = module(pred)
            diff = torch.square(normalize_tensor(target.permute(0, 3, 1, 2).contiguous()) - normalize_tensor(pred.permute(0, 3, 1, 2).contiguous()))
            diff = spatial_average(diff)
            diffs.append(diff)

        loss = diffs[0]
        for i in range(1, len(diffs)):
            loss += diffs[i]

        return loss


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)