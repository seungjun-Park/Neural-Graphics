import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from timm.models.layers import DropPath

from typing import Union, List, Tuple
from utils import conv_nd, to_2tuple, get_act, instantiate_from_config, group_norm
from modules.blocks import DownBlock, PatchMerging
from modules.blocks.attn_block import DoubleWindowSelfAttentionBlock, DoubleWindowCrossAttentionBlock
from modules.blocks.mlp import ConvMLP


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 mlp_ratio: float = 4.0,
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
        self.norm = group_norm(in_channels, num_groups=num_groups)
        self.norm2 = group_norm(in_channels, num_groups=num_groups)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = ConvMLP(
            in_channels=in_channels,
            embed_dim=int(in_channels * mlp_ratio),
            dropout=dropout,
            act=act,
            num_groups=num_groups,
            dim=dim,
            use_checkpoint=use_checkpoint
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv_nd(2, in_channels, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, attn_map = self.attn(x)
        h = self.act(self.norm(x + self.drop_path(h)))
        z = self.act(self.norm2(self.shortcut(h) + self.drop_path(self.mlp(h))))

        return z


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 out_channels: int = None,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_groups: int = 16,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 mlp_ratio: float = 4.0,
                 use_conv: bool = True,
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.self_attn = DoubleWindowSelfAttentionBlock(
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

        self.cross_attn = DoubleWindowCrossAttentionBlock(
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

        self.norm = group_norm(in_channels, num_groups=num_groups)
        self.norm2 = group_norm(in_channels, num_groups=num_groups)
        self.norm3 = group_norm(out_channels, num_groups)
        self.act = get_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = ConvMLP(
            in_channels=in_channels,
            embed_dim=int(in_channels * mlp_ratio),
            dropout=dropout,
            act=act,
            num_groups=num_groups,
            dim=dim,
            use_checkpoint=use_checkpoint
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv_nd(2, in_channels, out_channels, kernel_size=1, stride=1),
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h, attn_map = self.self_attn(x)
        h = self.act(self.norm(x + self.drop_path(h)))
        z, attn_map = self.cross_attn(context, h)
        z = self.act(self.norm2(h + self.drop_path(z)))
        z = self.act(self.norm3(self.shortcut(h) + self.drop_path(self.mlp(z))))

        return z


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 in_res: int = 512,
                 embed_dim: int = 32,
                 quant_dim: int = None,
                 hidden_dims: Union[List[int], Tuple[int]] = (),
                 num_blocks: Union[int, List[int], Tuple[int]] = (),
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 mlp_ratio: float = 4.0,
                 num_groups: int = 32,
                 pool_type: str = 'conv',
                 dim: int = 2,
                 use_conv: bool = True,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla'
                 ):
        super().__init__()

        assert num_heads != -1 or num_head_channels != -1
        if num_head_channels != -1:
            self.use_num_head_channels = True
        else:
            self.use_num_head_channels = False

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.embed = nn.Sequential(
            conv_nd(2, in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            group_norm(embed_dim, num_groups=num_groups),
            get_act(act),
        )

        in_ch = embed_dim
        cur_res = in_res // patch_size
        hidden_dims.append(1)  # dummy
        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks):
                self.encoder.append(
                    EncoderBlock(
                        in_channels=in_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=(in_ch // num_head_channels) if self.use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                        mlp_ratio=mlp_ratio
                    )
                )

                self.decoder.append(
                    DecoderBlock(
                        in_channels=in_ch,
                        in_res=cur_res,
                        window_size=window_size,
                        num_groups=num_groups,
                        num_heads=(in_ch // num_head_channels) if self.use_num_head_channels else num_heads,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        qkv_bias=qkv_bias,
                        bias=bias,
                        act=act,
                        use_conv=use_conv,
                        dim=dim,
                        use_checkpoint=use_checkpoint,
                        attn_mode=attn_mode,
                        mlp_ratio=mlp_ratio,
                    )
                )

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    PatchMerging(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=2,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        dim=dim,
                    )
                )

                self.encoder.append(
                    PatchMerging(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=2,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        dim=dim,
                    )
                )

                in_ch = out_ch
                cur_res //= 2

        quant_dim = in_ch if quant_dim is None else quant_dim
        self.quant_conv = conv_nd(dim, in_ch, quant_dim, kernel_size=1, stride=1)
        self.fc_w = nn.Parameter(torch.randn(1, quant_dim * cur_res ** 2))

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, training: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        context = self.embed(imgs)
        h = self.embed(edges)

        for i, (enc, dec) in enumerate(zip(self.encoder, self.decoder)):
            context = enc(context)
            if isinstance(dec, DecoderBlock):
                h = dec(h, context)
            else:
                h = dec(h)

        h = self.quant_conv(h)
        h = torch.flatten(h, start_dim=1)

        weights = self.fc_w
        direction = F.normalize(weights, dim=1)
        scale = torch.norm(self.fc_w, dim=1)
        h = h * scale
        if training:
            logits = (h.detach() * direction)
            dir = (h * direction.detach())
            out = {'logits': logits, 'dir': dir}
        else:
            logits = (h * direction)
            out = {'logits': logits}

        return out

