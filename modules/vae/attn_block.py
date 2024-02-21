import math

import torch
import torch.nn as nn

from modules.utils import activation_func, group_norm, conv_nd


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MHAttnBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self,
                 in_channels,
                 heads=8,
                 num_head_channels=-1,
                 ):

        super().__init__()
        self.channels = in_channels

        if num_head_channels != -1:
            self.num_heads = in_channels // num_head_channels
        else:
            self.num_heads = heads

        assert (
            in_channels % self.num_heads == 0
        ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {heads}"

        self.d_k = in_channels // heads
        self.norm = group_norm(in_channels)
        self.qkv = conv_nd(dim=1,
                           in_channels=in_channels,
                           out_channels=in_channels * 3,
                           kernel_size=1)

        # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ViTBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 heads=-1,
                 num_head_channels=-1,
                 dropout=0.,
                 attn_dropout=0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        assert heads != -1 or num_head_channels != -1
        if heads != -1:
            assert in_channels % heads == 0
            self.heads = heads
        else:
            assert in_channels % num_head_channels == 0
            self.heads = in_channels // num_head_channels


        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
        )
        self.mhattn_block = nn.MultiheadAttention(in_channels, num_heads=self.heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)

        h = self.proj_in(x)
        h = self.mhattn_block(h, h, h)
        h = self.dropout(h)
        h = h + x

        z = self.proj_out(h)
        z = z + h
        z = z.permute(0, 2, 1)
        z = z.reshape(b, c, *spatial)

        return z