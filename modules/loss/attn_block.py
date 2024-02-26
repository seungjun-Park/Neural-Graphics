import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 seq_length,
                 heads=-1,
                 num_head_channels=-1,
                 dropout=0.,
                 attn_dropout=0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        assert heads != -1 or num_head_channels != -1
        if heads != -1:
            assert in_channels % heads == 0
            self.heads = heads
        else:
            assert in_channels % num_head_channels == 0
            self.heads = in_channels // num_head_channels

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, in_channels).normal_(std=0.02))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
        )

        self.mhattn_block = nn.MultiheadAttention(in_channels, num_heads=self.heads, dropout=attn_dropout,
                                                  batch_first=True, bias=False)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels, bias=False),
        )

        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)

        x = x + self.pos_embedding

        h = self.proj_in(self.dropout1(x, self.dropout))
        h, _ = self.mhattn_block(h, h, h)
        h = self.dropout2(h)
        h = h + x

        z = self.proj_out(h)
        z = z + h
        z = self.ln(z)
        z = z.permute(0, 2, 1)
        z = z.reshape(b, -1, *spatial)

        return z