import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from utils import conv_nd, to_2tuple, get_act, instantiate_from_config
from modules.blocks import WindowAttnBlock, DownBlock, PatchMerging


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 in_res: int = 512,
                 embed_dim: int = 32,
                 hidden_dims: Union[List[int], Tuple[int]] = (),
                 attn_res: Union[List[int], Tuple[int]] = (),
                 patch_size: Union[int, List[int], Tuple[int]] = 4,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
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
        num_blocks
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


class EdgeImageDiscriminator(nn.Module):
    def __init__(self,
                 image_encoder_configs: dict,
                 edge_encoder_configs: dict,
                 in_channels: int = 3,
                 in_res: int = 512,
                 hidden_dims: Union[List[int], Tuple[int]] = (),
                 attn_res: Union[List[int], Tuple[int]] = (),
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path: float = 0.0,
                 qkv_bias: bool = True,
                 bias: bool = True,
                 act: str = 'relu',
                 dim: int = 2,
                 use_checkpoint: bool = False,
                 attn_mode: str = 'vanilla'
                 ):
        super().__init__()

        self.image_encoder = instantiate_from_config(image_encoder_configs).eval()
        self.edge_encoder = instantiate_from_config(edge_encoder_configs)

        in_ch = in_channels
        cur_res = in_res

        num_blocks = list(num_blocks)
        num_heads = list(num_heads)

        assert len(num_blocks) == 1 or len(num_blocks) >= len(hidden_dims)
        assert len(num_heads) == 1 or len(num_heads) >= len(hidden_dims)

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[0] if len(num_blocks) == 1 else num_blocks[i]):
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
                                num_heads=num_heads[0] if len(num_heads) == 1 else num_heads[i],
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
                self.blocks.append(PatchMerging(in_channels=in_ch))
                cur_res //= 2

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, training: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        feat_imgs = self.image_encoder(imgs)
        feat_edges = self.edge_encoder(edges)

        return self.disc(torch.cat([feat_imgs, feat_edges], dim=1), training=training)
