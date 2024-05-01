import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import to_2tuple, conv_nd, group_norm, instantiate_from_config, get_act
from modules.blocks import ResidualBlock, WindowAttnBlock, DownBlock, UpBlock, ScaledSkipBlock


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 out_channels: int = None,
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
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_addition_skip: bool = False,
                 ):
        super().__init__()

        self.num_blocks = num_blocks
        self.dim = dim
        self.mode = mode
        self.pool_type = pool_type
        self.use_conv = use_conv
        self.act = act
        self.num_groups = num_groups
        self.use_addition_skip = use_addition_skip

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        self.embed = nn.Sequential(
            conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
            group_norm(embed_dim, num_groups=1)
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = in_res

        self.encoder = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                down = list()
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

                skip_dims.append(in_ch)
                self.encoder.append(nn.Sequential(*down))

            if i != len(self.hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                skip_dims.append(in_ch)
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

        hidden_dims.pop()
        hidden_dims.insert(0, embed_dim)

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks):
                up = list()
                skip_dim = skip_dims.pop()
                up.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dim if not use_addition_skip else in_ch,
                        out_channels=in_ch if j != num_blocks - 1 else out_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                    )
                )

                in_ch = in_ch if j != num_blocks - 1 else out_ch

                if i in attn_res:
                    for k in range(2):
                        up.append(
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

                self.decoder.append(nn.Sequential(*up))

            if i != 0:
                skip_dim = skip_dims.pop()
                self.decoder.append(
                    UpBlock(in_ch + skip_dim if not use_addition_skip else in_ch,
                            out_channels=out_ch,
                            dim=dim,
                            mode=mode,
                            use_conv=use_conv
                    )
                )
                cur_res = int(cur_res * 2)

        skip_dim = skip_dims.pop()
        self.out = nn.Sequential(
            conv_nd(
                dim,
                in_ch + skip_dim if not use_addition_skip else in_ch,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = self.embed(x)
        hs.append(h)
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        for i, block in enumerate(self.middle):
            h = block(h)

        for i, block in enumerate(self.decoder):
            if self.use_addition_skip:
                h = (h + hs.pop()) * 0.5
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        if self.use_addition_skip:
            h = (h + hs.pop()) * 0.5
        else:
            h = torch.cat([h, hs.pop()], dim=1)
        h = self.out(h)

        return h


class UNet2Plus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return


class UNet3Plus(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 out_channels: int = None,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
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
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 use_lpf_conv: bool = True,
                 ):
        super().__init__()

        self.num_blocks = num_blocks
        self.dim = dim
        self.mode = mode
        self.pool_type = pool_type
        self.use_conv = use_conv
        self.act = act
        self.num_groups = num_groups

        self.in_channels = in_channels
        self.in_res = in_res
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims

        self.embed = nn.Sequential(
            conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
            group_norm(embed_dim, embed_dim)
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = in_res

        self.encoder = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.scaled_skip_blocks = nn.ModuleList()

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                down = list()
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
                        use_lpf_conv=use_lpf_conv,
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

                skip_dims.append(in_ch)
                self.encoder.append(nn.Sequential(*down))

            if i != len(self.hidden_dims) - 1:
                self.encoder.append(DownBlock(in_ch, dim=dim, use_conv=use_conv, pool_type=pool_type))
                skip_dims.append(in_ch)
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

        skip_dim = sum([i for i in skip_dims])

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            self.scaled_skip_blocks.append(
                ScaledSkipBlock(
                    level=i,
                    skip_dims=skip_dims,
                    unit=num_blocks,
                    dim=dim,
                    pool_type=pool_type,
                    mode=mode,
                    use_checkpoint=use_checkpoint,
                )
            )

            for j in range(num_blocks):
                up = list()
                up.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        act=act,
                        num_groups=num_groups,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                    )
                )

                in_ch = out_ch

                if i in attn_res:
                    for k in range(2):
                        up.append(
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

                self.decoder.append(nn.Sequential(*up))

            if i != 0:
                self.decoder.append(
                    UpBlock(in_ch + skip_dim,
                            out_channels=in_ch,
                            dim=dim,
                            mode=mode,
                            use_conv=use_conv
                            )
                )
                cur_res = int(cur_res * 2)

        self.out = nn.Sequential(
            conv_nd(dim, in_ch + skip_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = self.embed(x)
        hs.append(h)
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        for i, block in enumerate(self.middle):
            h = block(h)

        h_cats = []
        for i, block in enumerate(self.scaled_skip_blocks):
            h_cats.append(block(hs))

        for i, block in enumerate(self.decoder):
            h = torch.cat([h, h_cats[i // (self.num_blocks + 1)]], dim=1)
            h = block(h)

        h = torch.cat([h, h_cats[-1]], dim=1)
        h = self.out(h)

        return h
