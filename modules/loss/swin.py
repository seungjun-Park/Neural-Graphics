import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights


def get_pretrained_model(type):
    if type == 'swin-t':
        return swin_t(weights=Swin_T_Weights.DEFAULT).eval()
    elif type == 'swin-s':
        return swin_s(weights=Swin_S_Weights.DEFAULT).eval()
    elif type == 'swin-b':
        return swin_b(weights=Swin_B_Weights.DEFAULT).eval()
    elif type == 'swin-v2-t':
        return swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT).eval()
    elif type == 'swin-v2-s':
        return swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT).eval()
    elif type == 'swin-v2-b':
        return swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT).eval()
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

        swin = get_pretrained_model(self.type)
        features = swin.features.eval().requires_grad_(False).cuda()

        self.layers = [
            features[: 2].eval().requires_grad_(False),
            features[2: 4].eval().requires_grad_(False),
            features[4: 6].eval().requires_grad_(False),
            features[6: ].eval().requires_grad_(False),
        ]

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

            diff = normalize_tensor(target.contiguous()) - normalize_tensor(pred.contiguous())
            diff = torch.square(diff)
            diffs.append(diff)

        loss = diffs[0]
        for i in range(1, len(diffs)):
            loss += diffs[i]

        return loss



def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=3,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([1,2],keepdim=keepdim)

