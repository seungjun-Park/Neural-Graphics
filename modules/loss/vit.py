import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms.transforms import Resize


class ViTLoss(nn.Module):
    def __init__(self,
                 weights=ViT_B_16_Weights.DEFAULT
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        vit = vit_b_16(weights=weights).eval()

        self.conv_proj = vit.conv_proj
        self.encoder_blocks = nn.ModuleList()
        for i in range(4):
            self.encoder_blocks.append(vit.encoder.layers[3 * i: 3 * (i + 1)])

        for b in self.encoder_blocks:
            for p in b.parameters():
                p.requires_grad = False

        self.path_size = vit.patch_size
        self.image_size = vit.image_size
        self.hidden_dim = vit.hidden_dim
        self.class_token = vit.class_token

        self.pos_embedding = vit.encoder.pos_embedding

        self.transform = Resize(self.image_size, self.image_size)
        self.mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def preprocessing(self, x):
        x = (x - self.mean) / self.std
        x = self.transform(x)

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

    def forward(self, target, pred, feature_layers=[0, 1, 2, 3]):
        target = self.preprocessing(target)
        pred = self.preprocessing(pred)

        loss = 0.0
        for i, module in enumerate(self.encoder_blocks):
            target = module(target)
            pred = module(pred)

            if i in feature_layers:
                loss += F.l1_loss(target, pred)

        return loss

