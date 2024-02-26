import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        _log_api_usage_once(self)

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim, mlp_dim, dropout, *args, **kwargs):
        super().__init__(in_dim, [mlp_dim, in_dim], *args, activation_layer=nn.GELU, inplace=None, dropout=dropout, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 max_seq_len=5000,
                 heads=-1,
                 num_head_channels=-1,
                 dropout=0.,
                 attn_dropout=0.,
                 bias=True,
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

        self.pos_embedding = nn.Parameter(torch.empty(1, max_seq_len, in_channels, requires_grad=True).normal_(std=0.02))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
        )

        self.mhattn_block = nn.MultiheadAttention(in_channels, num_heads=self.heads, dropout=attn_dropout,
                                                  batch_first=True, bias=bias)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(in_channels),
            MLPBlock(in_channels, in_channels * 2, dropout=dropout, bias=bias)
        )

        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = x.permute(0, 2, 1)

        x = x + self.pos_embedding[:, : x.shpae[1], :]

        h = self.proj_in(self.dropout1(x))
        h, _ = self.mhattn_block(h, h, h)
        h = self.dropout2(h)
        h = h + x

        z = self.proj_out(h)
        z = z + h
        z = self.ln(z)
        z = z.permute(0, 2, 1)
        z = z.reshape(b, -1, *spatial)

        return z