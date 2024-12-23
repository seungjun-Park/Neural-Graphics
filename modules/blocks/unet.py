from collections import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
from timm.models.layers import DropPath
from omegaconf import ListConfig

from utils import to_2tuple, conv_nd, group_norm, instantiate_from_config, get_act, checkpoint, to_ntuple, freq_mask
from modules.blocks.patches import PatchMerging, PatchExpanding
from modules.blocks.mlp import MLP, ConvMLP
from modules.blocks.attn_block import AttentionBlock
from modules.blocks.res_block import ResidualBlock
from modules.blocks.down import DownBlock
from modules.blocks.up import UpBlock
from modules.blocks.deform_conv import deform_conv_nd
from modules.blocks.norm import GlobalResponseNorm


class UnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 window_size: int = 7,
                 shift_size: int = 0,
                 num_groups: int = 16,
                 num_heads: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 qk_scale: float = None,
                 mlp_ratio: float = 4.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_type: str = 'mha',
                 scale_factor: int = 4,
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.attn = AttentionBlock(
            in_channels=in_channels,
            in_res=in_res,
            window_size=window_size,
            shift_size=shift_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            attn_type=attn_type,
            dim=dim,
            scale_factor=scale_factor,
            use_checkpoint=use_checkpoint,
        )

        self.mlp = ConvMLP(
            in_channels=in_channels,
            embed_dim=int(in_channels * mlp_ratio),
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            dim=dim,
            use_checkpoint=use_checkpoint
        )

        self.norm = group_norm(in_channels, num_groups=1)
        self.norm2 = group_norm(out_channels, num_groups=1)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = conv_nd(dim, in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x)
        h = self.act(self.norm(x + self.drop_path(h)))
        h = self.act(self.norm2(self.shortcut(h) + self.drop_path(self.mlp(h))))

        return h


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 attn_types: Union[List[str], Tuple[str]],
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
                 proj_bias: bool = True,
                 mlp_ratio: float = 4.0,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        out_channels = out_channels if out_channels else in_channels

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
                group_norm(embed_dim, num_groups=1),
            )
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]
        cur_res = in_res

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                self.encoder.append(
                    UnetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        num_groups=num_groups,
                        num_heads=in_ch // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        mlp_ratio=mlp_ratio,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_type=attn_types[i],
                    )
                )

                in_ch = out_ch
                skip_dims.append(in_ch)

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        scale_factor=2,
                        dim=2,
                        pool_type=pool_type,
                        use_checkpoint=use_checkpoint
                    )
                )
                cur_res //= 2

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                skip_dim = skip_dims.pop()
                self.decoder.append(
                    UnetBlock(
                        in_channels=in_ch + skip_dim,
                        out_channels=out_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        num_groups=num_groups,
                        num_heads=(in_ch + skip_dim) // num_head_channels if use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        mlp_ratio=mlp_ratio,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_type=attn_types[i],
                    )
                )
                in_ch = out_ch

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch,
                        scale_factor=2,
                        mode=mode,
                        use_checkpoint=use_checkpoint
                    )
                )
                cur_res *= 2

        self.out = nn.Sequential(
            conv_nd(
                dim,
                in_ch + skip_dims.pop(),
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
            if not isinstance(block, DownBlock):
                hs.append(h)

        for i, block in enumerate(self.decoder):
            if isinstance(block, UnetBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)
        h = torch.cat([h, hs.pop()], dim=1)
        return self.out(h)


class DeformableUNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 act: str = 'relu',
                 use_conv: bool = True,
                 num_groups: int = 1,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 modulation_type: str = 'none',
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        out_channels = out_channels if out_channels is not None else in_channels

        self.res_block = ResidualBlock(
            in_channels,
            out_channels,
            dropout=dropout,
            drop_path=drop_path,
            act=act,
            dim=dim,
            num_groups=num_groups,
            use_checkpoint=False,
            use_conv=use_conv,
        )

        if deformable_group_channels is not None:
            deformable_groups = out_channels // deformable_group_channels

        self.dcn = nn.Sequential(
            group_norm(out_channels, num_groups=num_groups),
            get_act(act),
            deform_conv_nd(
                dim,
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=deformable_groups,
                offset_scale=offset_scale,
                fix_center=fix_center,
                modulation_type=modulation_type,
                dw_kernel_size=dw_kernel_size,
            ),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == b, c, *...
        x = self.res_block(x)
        h = self.dcn(x)

        h = self.drop_path(h) + x

        return h


class DeformableUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 num_groups: int = 1,
                 deformable_groups: int = 1,
                 deformable_group_channels: int = None,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 modulation_type: str = 'none',
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 mode: str = 'nearest',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 use_last_layer: bool = True
                 ):
        super().__init__()

        out_channels = out_channels if out_channels else in_channels

        self.dim = dim
        self.hidden_dims = hidden_dims
        self.num_groups = num_groups
        self.act = act

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                conv_nd(
                    dim,
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

        in_ch = embed_dim
        skip_dims = [embed_dim]

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, abc.Iterable) else num_blocks):
                self.encoder.append(
                    DeformableUNetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        fix_center=fix_center,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )
                in_ch = out_ch
                skip_dims.append(in_ch)

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        dim=2,
                        pool_type=pool_type,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )
                skip_dims.append(in_ch)

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, ListConfig) else num_blocks):
                in_ch = in_ch + skip_dims.pop()
                self.decoder.append(
                    DeformableUNetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        drop_path=drop_path,
                        act=act,
                        use_conv=use_conv,
                        num_groups=num_groups,
                        deformable_groups=deformable_groups,
                        deformable_group_channels=deformable_group_channels,
                        offset_scale=offset_scale,
                        fix_center=fix_center,
                        modulation_type=modulation_type,
                        dw_kernel_size=dw_kernel_size,
                        dim=dim,
                        use_checkpoint=use_checkpoint
                    )
                )
                in_ch = out_ch

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        scale_factor=2,
                        mode=mode,
                        use_checkpoint=use_checkpoint,
                        num_groups=num_groups,
                    )
                )

        if use_last_layer:
            self.out = nn.Sequential(
                group_norm(in_ch, num_groups=num_groups),
                get_act(act),
                conv_nd(
                    dim,
                    in_ch,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        else:
            self.out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            hs.append(h)

        for i, block in enumerate(self.decoder):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h)

        if self.out is not None:
            h = self.out(h)

        return h
