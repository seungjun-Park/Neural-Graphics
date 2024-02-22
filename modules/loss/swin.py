import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, swin_t, swin_s, swin_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights


class SwinLoss(nn.Module):
    def __init__(self,
                 type='swin-t',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.type = type.lower()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def get_pretrained_model(self):
        if self.type == 'swin-t':
            return swin_t(weights=Swin_T_Weights.DEFAULT)
        elif self.type == 'swin-s':
            return swin_s(weights=Swin_S_Weights.DEFAULT)
        elif self.type == 'swin-b':
            return swin_b(weights=Swin_B_Weights.DEFAULT)
        elif self.type == 'swin-v2-t':
            return swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        elif self.type == 'swin-s':
            return swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        elif self.type == 'swin-b':
            return swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        else:
            NotImplementedError(f'{self.type} is not available.')

    def preprocessing(self, x):
        x = (x - self.mean) / self.std
        x = F.interpolate(x, mode='bilinear', size=(self.image_size, self.image_size), align_corners=False, antialias=True)

        b, c, h, w = x.shape
        n_h = h // self.path_size
        n_w = w // self.path_size

        x = self.conv_proj(x)
        x = x.reshape(b, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        batch_class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding

        return x

    def forward(self, target, pred):
        b, c, h, w = target.shape

        target = self.preprocessing(target)
        pred = self.preprocessing(pred)

        l1_loss = 0.0
        for i, module in enumerate(self.encoder_blocks):
            target = module(target)
            pred = module(pred)

            l1_loss += F.l1_loss(target, pred)

        return l1_loss
