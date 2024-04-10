import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import hinge_d_loss, weights_init, vanilla_d_loss, NLayerDiscriminator, adopt_weight, LPIPS
from models.gan.discriminator import Discriminator


class EdgePerceptualLoss(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, disc_num_layers=3, disc_in_channels=1, disc_embed_dim=64,
                 l1_weight=1.0, perceptual_weight=1.0, disc_factor=1.0, disc_weight=1e-4, disc_loss="hinge"):

        super().__init__()
        disc_loss = disc_loss.lower()
        assert disc_loss in ["hinge", "vanilla"]
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.l1_weight = l1_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = Discriminator(net=self.perceptual_loss.net,
                                           in_channels=disc_in_channels,
                                           embed_dim=disc_embed_dim,
                                           num_layers=disc_num_layers,
                                           ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.disc_weight).detach()
        return d_weight

    def forward(self, inputs, target, cond, optimizer_idx, global_step, last_layer, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - target.contiguous()) * self.l1_weight

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.repeat(1, 3, 1, 1).contiguous(), target.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(target.repeat(1, 3, 1, 1).contiguous(), cond.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                if self.training:
                    g_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                else:
                    g_weight = torch.tensor(0.0)
            else:
                g_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = rec_loss + g_loss * g_weight * disc_factor

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.repeat(1, 3, 1, 1).contiguous().detach(), cond.contiguous().detach())
            logits_fake = self.discriminator(target.repeat(1, 3, 1, 1).contiguous().detach(), cond.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            return d_loss, log