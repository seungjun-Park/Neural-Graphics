import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight, LPIPS
from models.gan.discriminator import Discriminator

class EdgePerceptualLoss(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=1, disc_embed_dim=64,
                 l1_weight=1.0, perceptual_weight=1.0, d_weight=1.0, g_weight=1e-4, disc_loss="hinge"):

        super().__init__()
        disc_loss = disc_loss.lower()
        assert disc_loss in ["hinge", "vanilla"]
        self.d_weight = d_weight
        self.g_weight = g_weight
        self.l1_weight = l1_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = Discriminator(net=self.perceptual_loss.net,
                                           in_channels=disc_in_channels,
                                           embed_dim=disc_embed_dim,
                                           num_layers=disc_num_layers,
                                           ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def to(self, device=None, *args, **kwargs):
        super().to(device=device, *args, **kwargs)
        self.perceptual_loss.to(device, *args, **kwargs)

    def forward(self, inputs, target, cond, optimizer_idx, global_step, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - target.contiguous()) * self.l1_weight

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.repeat(1, 3, 1, 1).contiguous(), target.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(target.repeat(1, 3, 1, 1).contiguous(), cond.contiguous())
            g_loss = -torch.mean(logits_fake)

            g_weight = adopt_weight(self.g_weight, global_step, threshold=self.discriminator_iter_start)
            loss = rec_loss + g_loss * g_weight

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.repeat(1, 3, 1, 1).contiguous().detach(), cond.contiguous().detach())
            logits_fake = self.discriminator(target.repeat(1, 3, 1, 1).contiguous().detach(), cond.contiguous().detach())

            d_loss = self.d_weight * self.disc_loss(logits_real, logits_fake)

            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log