import torch

import torch.nn as nn
from typing import Tuple, List, Union

from modules.blocks import ResidualBlock, WindowAttnBlock, DownBlock, UpBlock
from utils import conv_nd, group_norm, to_2tuple


class SwinEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 embed_dim: int = 16,
                 attn_res: Union[List[int], Tuple[int]] = (2, 3),
                 num_blocks: int = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'max',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        self.embed = nn.Sequential(
                conv_nd(dim, in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
                group_norm(embed_dim, num_groups=1),  # equal to layer norm
            )
        self.encoder = nn.ModuleList()
        self.middle = nn.ModuleList()

        in_ch = embed_dim
        cur_res = in_res // patch_size

        for i, out_ch in enumerate(hidden_dims):
            down = list()
            for j in range(num_blocks):
                down.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        use_conv=use_conv,
                    )
                )
                in_ch = out_ch

                if i in attn_res:
                    for k in range(2):
                        down.append(
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

            self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                cur_res //= 2

        for i in range(num_blocks):
            self.middle.append(
                nn.Sequential(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        use_conv=use_conv,
                    ),
                    *[
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
                        for k in range(2)
                    ]
                )
            )

    def forward(self, x: torch.Tensor, use_deep_supervision: bool = False, **ignored_kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        hs = []
        h = self.embed(x)

        for i, module in enumerate(self.encoder):
            h = module(h)
            if i % 2 == 0:
                hs.append(h)

        for i, module in enumerate(self.middle):
            h = module(h)

        if use_deep_supervision:
            hs.append(h)
            return hs

        return h
