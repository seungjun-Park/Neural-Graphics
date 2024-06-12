import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config, to_2tuple, to_3tuple, conv_nd, get_act, group_norm, normalize_img
from modules.sequential import AttentionSequential
from utils.loss import CriterionBlock


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

        self.encoder_img = instantiate_from_config(encoder_img_config).eval()
        self.encoder_edge = instantiate_from_config(encoder_edge_config).eval()

        self.criterion_blocks = nn.ModuleList()
        encoder_params = encoder_img_config['params']
        cur_res = encoder_params['in_res']
        for i, hidden_dim in enumerate(encoder_params['hidden_dims']):
            self.criterion_blocks.append(
                CriterionBlock(
                    in_channels=hidden_dim,
                    in_res=cur_res,
                    num_heads=num_heads[i] if isinstance(num_heads, ListConfig) else num_heads,
                    window_size=window_size,
                    bias=bias,
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

            cur_res //= 2

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

    def forward(self, img: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        feat_imgs, _ = self.encoder_img.feature_extract(img)
        feat_edges, _ = self.encoder_edge.feature_extract(edge)
        criterion_feats = []
        for i, (feat_img, feat_edge, criterion_block) in enumerate(zip(feat_imgs, feat_edges, self.criterion_blocks)):
            criterion_feat = criterion_block(self._normalize_feature(feat_edge), self._normalize_feature(feat_img))
            criterion_feats.append(self._spatial_average(criterion_feat))

        criterion = criterion_feats[0]
        for i in range(1, len(criterion_feats)):
            criterion += criterion_feats[i]

        return F.sigmoid(criterion)

    def training_step(self, batch, batch_idx):
        img, edge, label = batch

        dist = self(img, edge).mean()

        loss = dist * label

        self.log('train/loss', loss, logger=True, rank_zero_only=True)
        self.log('train/dist', dist.detach().mean(), logger=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, label = batch

        dist = self(img, edge)

        loss = dist * label

        self.log('val/loss', loss, logger=True, rank_zero_only=True)
        self.log('val/dist', dist.detach().mean(), logger=True, rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        params = list(self.criterion_blocks.parameters())

        opt = torch.optim.AdamW(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(0.5, 0.9)
                                )

        return [opt]