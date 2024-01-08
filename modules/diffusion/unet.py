import torch as th
import torch.nn as nn
import torch.nn.functional as F

from . import TimestepBlock, TimestepEmbedSequential
from modules.diffusion.res_block import ResBlock
from modules.diffusion.attn import AttentionBlock
from modules.diffusion.embedding import PositionalEmbedding, GaussianFourierProjection, timestep_embedding
from modules.diffusion.down import Downsample
from modules.diffusion.up import Upsample
from utils.activation import Sine

from modules.diffusion.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
)


class UNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads=-1,
            num_head_channels=-1,
            use_sine_act=False,
            use_fourier=False,
            use_biggan_res=False,
            ):
            super().__init__()

            act = Sine if use_sine_act else nn.SiLU

            self.in_channels = in_channels
            self.model_channels = model_channels
            self.out_channels = out_channels
            self.num_res_blocks = num_res_blocks
            self.attention_resolutions = attention_resolutions
            self.dropout = dropout
            self.channel_mult = channel_mult
            self.num_classes = num_classes
            self.use_checkpoint = use_checkpoint
            self.dtype = th.float32
            self.num_heads = num_heads
            self.num_head_channels = num_head_channels
            self.use_biggan_res = use_biggan_res

            self.pos_enc = GaussianFourierProjection(self.model_channels) if use_fourier else PositionalEmbedding(self.model_channels)

            time_embed_dim = model_channels * 4
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                act(),
                linear(time_embed_dim, time_embed_dim),
            )

            if self.num_classes is not None:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)

            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            self._feature_size = model_channels
            input_block_chans = [model_channels]
            ch = model_channels
            ds = 1
            for level, mult in enumerate(channel_mult):
                for _ in range(num_res_blocks):
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_sine_act=use_sine_act,
                        )
                    ]
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        if self.num_heads == -1:
                            num_heads = ch // num_head_channels
                        else:
                            num_heads = self.num_heads

                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                            )
                        )
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch
                    input_block_chans.append(ch)
                if level != len(channel_mult) - 1:
                    out_ch = ch
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                down=True,
                                use_sine_act=use_sine_act,
                            )
                            if self.use_biggan_res else
                            Downsample(ch, use_conv=True, out_channels=out_ch)
                        )
                    )
                    ch = out_ch
                    input_block_chans.append(ch)
                    ds *= 2
                    self._feature_size += ch

            if self.num_heads == -1:
                num_heads = ch // num_head_channels
            else:
                num_heads = self.num_heads

            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_sine_act=use_sine_act,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_sine_act=use_sine_act,
                ),
            )
            self._feature_size += ch

            self.output_blocks = nn.ModuleList([])

            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_sine_act=use_sine_act,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if self.num_heads == -1:
                            num_heads = ch // num_head_channels
                        else:
                            num_heads = self.num_heads
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                            )
                        )
                    if level and i == num_res_blocks:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                up=True,
                                use_sine_act=use_sine_act,
                            )
                            if use_biggan_res else
                            Upsample(ch, use_conv=True, out_channels=out_ch)
                        )
                        ds //= 2

                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch

            self.out = nn.Sequential(
                normalization(ch),
                act(),
                conv_nd(dims, ch, out_channels, 3, padding=1),
            )

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = self.pos_enc(timesteps)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h.type(x.dtype)
        return self.out(h)