import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import conv_nd, to_2tuple
from modules.blocks import WindowAttnBlock, SpectralNorm


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 in_res: int = 512,
                 hidden_dims: Union[List[int], Tuple[int]] = (),
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
                 dim: int = 2,
                 use_checkpoint: bool = False,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_ch = in_channels
        cur_res = in_res
        for i, out_ch in enumerate(hidden_dims):
            self.blocks.append(
                nn.Sequential(
                    SpectralNorm(conv_nd(dim, in_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                    nn.LeakyReLU(0.1, True)
                )
            )

            in_ch = out_ch
            cur_res = cur_res // 2

            self.blocks.append(
                nn.Sequential(
                    WindowAttnBlock(
                        in_channels=in_ch,
                        in_res=to_2tuple(cur_res),
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=bias,
                        drop=dropout,
                        attn_drop=attn_dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=False,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                    ),
                    WindowAttnBlock(
                        in_channels=in_ch,
                        in_res=to_2tuple(cur_res),
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        window_size=window_size,
                        shift_size=window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=bias,
                        drop=dropout,
                        attn_drop=attn_dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=False,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                    ),
                )
            )
        self.blocks.append(conv_nd(dim, in_ch, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
