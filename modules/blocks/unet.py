import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from timm.models.layers import DropPath
from omegaconf import ListConfig

from utils import to_2tuple, conv_nd, group_norm, instantiate_from_config, get_act
from modules.blocks.patches import PatchMerging, PatchExpanding
from modules.blocks.mlp import MLP, ConvMLP
from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock, SelfAttentionBlock
from modules.blocks.res_block import ResidualBlock
from modules.blocks.down import DownBlock
from modules.blocks.up import UpBlock
from modules.blocks.positional_encoding import PositionalEncoding
from modules.sequential import AttentionSequential


class UnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 mlp_ratio: float = 4.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.attn = DoubleWindowSelfAttentionBlock(
            in_channels=in_channels,
            in_res=to_2tuple(in_res),
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            proj_bias=bias,
            dropout=attn_dropout,
            use_checkpoint=use_checkpoint,
            attn_mode=attn_mode,
            dim=dim
        )

        self.res_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            dim=dim,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint,
            use_conv=use_conv,
        )

        self.norm = group_norm(in_channels, num_groups=num_groups)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, attn_map = self.attn(x)
        h = self.act(self.norm(x + self.drop_path(h)))
        h = self.res_block(h)

        return h


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 out_channels: int = None,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 mlp_ratio: float = 4.0,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        assert num_head_channels != -1 or num_heads != -1

        if num_head_channels != -1:
            use_num_head_channels = True
        else:
            use_num_head_channels = False

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
                group_norm(embed_dim, num_groups=num_groups),
                get_act(act)
            )
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = in_res

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                down = list()
                down.append(
                    UnetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=in_ch // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        mlp_ratio=mlp_ratio,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )

                in_ch = out_ch
                skip_dims.append(in_ch)
                self.encoder.append(nn.Sequential(*down))

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        scale_factor=2,
                        dim=2,
                        num_groups=num_groups,
                        pool_type=pool_type
                    )
                )
                skip_dims.append(in_ch)
                cur_res //= 2

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                up = list()
                skip_dim = skip_dims.pop()
                up.append(
                    UnetBlock(
                        in_channels=in_ch + skip_dim,
                        out_channels=out_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=(in_ch + skip_dim) // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        mlp_ratio=mlp_ratio,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                    )
                )
                in_ch = out_ch
                self.decoder.append(nn.Sequential(*up))

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        scale_factor=2,
                        mode=mode,
                        num_groups=num_groups
                    )
                )
                cur_res *= 2

        skip_dim = skip_dims.pop()

        self.attn_out = DoubleWindowSelfAttentionBlock(
            in_channels=in_ch + skip_dim,
            in_res=to_2tuple(in_res),
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            proj_bias=bias,
            dropout=attn_dropout,
            use_checkpoint=use_checkpoint,
            attn_mode=attn_mode,
            dim=dim
        )

        self.out = nn.Sequential(
            group_norm(in_ch + skip_dim, num_groups=num_groups),
            get_act(act),
            conv_nd(
                dim,
                in_ch + skip_dim,
                out_channels,
                kernel_size=1,
                stride=1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        for i, block in enumerate(self.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        h = torch.cat([h, hs.pop()], dim=1)
        h, attn_map = self.attn_out(h)
        return self.out(h)

