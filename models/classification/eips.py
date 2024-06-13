import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.sequential import AttentionSequential
from modules.blocks.attn_block import ResidualSelfAttentionBlock, ResidualCrossAttentionBlock
from modules.blocks.down import DownBlock


class EIPS(pl.LightningModule):
    def __init__(self,
                 encoder_img_config: DictConfig,
                 encoder_edge_config: DictConfig,
                 num_heads: Union[int, List[int], Tuple[int]] = 8,
                 window_size: Union[int, List[int], Tuple[int]] = 7,
                 bias: bool = True,
                 dropout: float = 0.,
                 attn_dropout: float = 0.,
                 drop_path: float = 0.0,
                 act: str = 'relu',
                 num_groups: int = 32,
                 use_conv: bool = True,
                 attn_mode: str = 'vanilla',
                 pool_type: str = 'max',
                 dim: int = 2,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval

        model_img = instantiate_from_config(encoder_img_config).eval()
        model_edge = instantiate_from_config(encoder_edge_config).eval()

        self.encoder_img = nn.ModuleList([model_img.net.embed, *model_img.net.encoder])
        self.encoder_edge = nn.ModuleList([model_edge.net.embed, *model_edge.net.encoder])

        self.cross_attn_blocks = nn.ModuleList()
        self.similarity_blocks = nn.ModuleList()
        encoder_params = encoder_img_config['params']['net_config']['params']
        self.cur_res = encoder_params['in_res']
        hidden_dims = encoder_params['hidden_dims']
        in_ch = 0
        for i, hidden_dim in enumerate(hidden_dims):
            self.cross_attn_blocks.append(
                ResidualCrossAttentionBlock(
                    in_channels=hidden_dim,
                    in_res=self.cur_res,
                    num_heads=num_heads[i] if isinstance(num_heads, ListConfig) else num_heads,
                    window_size=window_size,
                    proj_bias=bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                    act=act,
                    num_groups=num_groups,
                    use_conv=use_conv,
                    attn_mode=attn_mode,
                    dim=dim,
                )
            )

            self.similarity_blocks.append(
                AttentionSequential(
                    ResidualSelfAttentionBlock(
                        in_channels=in_ch + hidden_dim,
                        in_res=self.cur_res,
                        out_channels=hidden_dim,
                        num_heads=num_heads[i] if isinstance(num_heads, ListConfig) else num_heads,
                        window_size=window_size,
                        proj_bias=bias,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=drop_path,
                        act=act,
                        num_groups=num_groups,
                        use_conv=use_conv,
                        attn_mode=attn_mode,
                        dim=dim
                    )
                )
            )

            in_ch = hidden_dim

            if i != len(hidden_dims) - 1:
                self.similarity_blocks.append(
                    DownBlock(in_ch, dim=dim, num_groups=num_groups, pool_type=pool_type)
                )
                self.cur_res //= 2

        self.logit = nn.Linear(in_features=int(hidden_dim * (self.cur_res ** 2)), out_features=1)

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def train(self, mode: bool = True):
        super().train(mode)
        for param in self.parameters():
            param.requires_grad = True

        return self

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def _spatial_average(self, x: torch.Tensor, keepdim=True):
        return x.mean([2, 3], keepdim=keepdim)

    def _normalize_feature(self, feature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(feature ** 2, dim=1, keepdim=True))
        return feature / (norm_factor + eps)

    @torch.no_grad()
    def _feature_extract(self, img: torch.Tensor, edge: torch.Tensor) -> List[torch.Tensor]:
        feats_img, feats_edge = [], []
        feat_img = img
        feat_edge = edge

        for i, (module_img, module_edge) in enumerate(zip(self.encoder_img, self.encoder_edge)):
            if isinstance(module_img, AttentionSequential):
                feat_img, attn_map_img = module_img(feat_img)
                feats_img.append(feat_img)
            else:
                feat_img = module_img(feat_img)

            if isinstance(module_edge, AttentionSequential):
                feat_edge, attn_map_edge = module_img(feat_edge)
                feats_edge.append(feat_edge)
            else:
                feat_edge = module_img(feat_edge)

        return feats_img, feats_edge

    def forward(self, img: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        feat_imgs, feat_edges = self._feature_extract(img, edge)
        cross_attn_feats = []
        for i, (feat_img, feat_edge, cross_attn_block) in enumerate(zip(feat_imgs, feat_edges, self.cross_attn_blocks)):
            cross_attn_feats.append(cross_attn_block(self._normalize_feature(feat_edge), self._normalize_feature(feat_img))[0])

        feat = cross_attn_feats.pop(0)
        for i, module in enumerate(self.similarity_blocks):
            if isinstance(module, AttentionSequential):
                if i != 0:
                    feat = torch.cat([feat, cross_attn_feats.pop(0)], dim=1)
                feat, attn_map = module(feat)
            else:
                feat = module(feat)

        feat = torch.flatten(feat, start_dim=1)
        feat = self.logit(feat)

        return F.sigmoid(feat)

    def training_step(self, batch, batch_idx):
        img, edge, label = batch

        similarity = self(img, edge)

        loss = ((similarity - 0.5) * label).mean()

        self.log('train/loss', loss, logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, label = batch

        similarity = self(img, edge)

        loss = ((similarity - 0.5) * label).mean()

        self.log('val/loss', loss, logger=True, rank_zero_only=True)

    @torch.no_grad()
    def log_img(self, img, edge):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/img', torch.clamp(img[0], min=0.0, max=1.0), self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/edge', torch.clamp(edge[0], min=0.0, max=1.0), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(list(self.cross_attn_blocks.parameters()) +
                                list(self.similarity_blocks.parameters()),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        return [opt]