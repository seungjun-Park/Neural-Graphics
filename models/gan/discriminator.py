import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import conv_nd, to_2tuple, get_act, instantiate_from_config, group_norm
from modules.blocks import DownBlock, PatchMerging
from modules.blocks.attn_block import ResidualSelfAttentionBlock, ResidualCrossAttentionBlock


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
                            kernel_size=5,
                            stride=1,
                            padding=2,
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
                self.blocks.append(DownBlock(in_channels=in_ch, dim=dim, pool_type=pool_type))
                cur_res //= 2

    def forward(self, x: torch.Tensor, training: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
        h = self.quant_conv(h)
        direction = torch.norm(self.fc_w, dim=1, p=2.0, keepdim=True)
        scale = torch.norm(self.fc_w, dim=1, keepdim=True)
        h = h * scale
        if training:
            logits = (h.detach() * direction)
            dir = (h * direction.detach())
            out = {'logits': logits, 'dir': dir}
        else:
            logits = (h * direction)
            out = logits

        return out


class SwinDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_res: Union[int, List[int], Tuple[int]],
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 embed_dim: int = 16,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 quant_dim: int = 8,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 bias: bool = True,
                 num_groups: int = 32,
                 act: str = 'relu',
                 use_conv: bool = True,
                 pool_type: str = 'conv',
                 dim: int = 2,
                 use_checkpoint: bool = True,
                 attn_mode: str = 'cosine',
                 ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                nn.utils.spectral_norm(
                    conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
                ),
                get_act(act)
            )
        )

        self.decoder.append(
            nn.Sequential(
                nn.utils.spectral_norm(
                    conv_nd(dim, in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
                ),
                get_act(act)
            )
        )

        in_ch = embed_dim
        cur_res = in_res

        for i, out_ch in enumerate(hidden_dims):
            self.encoder.append(
                ResidualSelfAttentionBlock(
                    in_channels=in_ch,
                    in_res=cur_res,
                    out_channels=out_ch,
                    num_heads=num_heads,
                    window_size=window_size,
                    proj_bias=bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                    act=act,
                    num_groups=num_groups,
                    use_conv=use_conv,
                    dim=dim,
                    use_checkpoint=use_checkpoint,
                    attn_mode=attn_mode,
                )
            )

            self.decoder.append(
                ResidualSelfAttentionBlock(
                    in_channels=in_ch,
                    in_res=cur_res,
                    out_channels=out_ch,
                    num_heads=num_heads,
                    window_size=window_size,
                    proj_bias=bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                    act=act,
                    num_groups=num_groups,
                    use_conv=use_conv,
                    dim=dim,
                    use_checkpoint=use_checkpoint,
                    attn_mode=attn_mode,
                )
            )

            self.decoder.append(
                ResidualCrossAttentionBlock(
                    in_channels=out_ch,
                    in_res=cur_res,
                    out_channels=out_ch,
                    num_heads=num_heads,
                    window_size=window_size,
                    proj_bias=bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                    act=act,
                    num_groups=num_groups,
                    use_conv=use_conv,
                    dim=dim,
                    use_checkpoint=use_checkpoint,
                    attn_mode=attn_mode,
                )
            )

            in_ch = out_ch

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(in_channels=in_ch, dim=dim, num_groups=num_groups, pool_type=pool_type),
                )
                self.decoder.append(
                    DownBlock(in_channels=in_ch, dim=dim, num_groups=num_groups, pool_type=pool_type),
                )

                cur_res //= 2

        self.quant = conv_nd(dim, in_ch, quant_dim, kernel_size=1, stride=1)

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        feat_imgs = []
        feat_img = imgs
        feat_edge = edges
        for i, module in enumerate(self.encoder):
            if isinstance(module, ResidualSelfAttentionBlock):
                feat_img, attn_map = module(feat_img)
                feat_imgs.append(feat_img)
            else:
                feat_img = module(feat_img)

        for i, module in enumerate(self.decoder):
            if isinstance(module, ResidualSelfAttentionBlock):
                feat_edge, attn_map = module(feat_edge)
            if isinstance(module, ResidualCrossAttentionBlock):
                feat_edge, attn_map = module(feat_edge, feat_imgs.pop(0))
            else:
                feat_edge = module(feat_edge)

        return self.quant(feat_edge)
