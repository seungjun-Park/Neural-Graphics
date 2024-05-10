import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import conv_nd, to_2tuple, get_act
from modules.blocks import WindowAttnBlock, DownBlock


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 in_res: int = 512,
                 embed_dim: int = 32,
                 hidden_dims: Union[List[int], Tuple[int]] = (),
                 attn_res: Union[List[int], Tuple[int]] = (),
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 pool_type: str = 'conv',
                 dim: int = 2,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla'
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_ch = in_channels
        cur_res = in_res
        self.embed = nn.utils.spectral_norm(conv_nd(dim, in_ch, embed_dim, kernel_size=patch_size, stride=patch_size))
        in_ch = embed_dim
        cur_res //= patch_size

        for i, out_ch in enumerate(hidden_dims):
            self.blocks.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        conv_nd(
                            dim,
                            in_ch,
                            out_ch,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    ),
                    get_act(act)
                )
            )

            in_ch = out_ch

            if i in attn_res:
                for k in range(2):
                    self.blocks.append(
                        WindowAttnBlock(
                            in_channels=in_ch,
                            in_res=to_2tuple(cur_res),
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            window_size=window_size,
                            shift_size=0 if k % 2 == 0 else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            proj_bias=bias,
                            drop=dropout,
                            attn_drop=attn_dropout,
                            drop_path=drop_path,
                            act=act,
                            use_checkpoint=use_checkpoint,
                            attn_mode=attn_mode,
                        )
                    )

            if i != len(hidden_dims) - 1:
                self.blocks.append(DownBlock(in_ch, dim=dim, pool_type=pool_type))
                cur_res //= 2

        self.fc_w = nn.Parameter(torch.randn(1, in_ch * cur_res * cur_res), requires_grad=True)

    def forward(self, x: torch.Tensor, training: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
        h = torch.flatten(h, start_dim=1)
        direction = F.normalize(self.fc_w, dim=1)
        scale = torch.norm(self.fc_w, dim=1).unsqueeze(1)
        h = h * scale
        if training:
            logits = (h.detach() * direction).sum(dim=1)
            dir = (h * direction.detach()).sum(dim=1)
            out = {'logits': logits, 'dir': dir}
        else:
            logits = (h * direction).sum(dim=1)
            out = logits

        return out
