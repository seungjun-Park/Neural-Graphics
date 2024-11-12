import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

import custom_op
from utils import to_1tuple, to_2tuple, to_3tuple, multiply_integers, group_norm, get_act


class DeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups_per_groups: int = 1,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 7,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 3,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.fix_center = fix_center

        kernel_sizes = multiply_integers(self.kernel_size)

        if fix_center:
            assert kernel_sizes % 2 != 0

        if fix_center and kernel_sizes == 1:
            self.offset_field = None
        else:
            self.offset_field = nn.Conv1d(
                in_channels,
                groups * deformable_groups_per_groups * (kernel_sizes - fix_center),
                kernel_size=kernel_size_off,
                stride=stride_off,
                padding=padding_off,
                dilation=dilation_off,
                groups=groups * deformable_groups_per_groups,
            )

        self.attn_mask = nn.Conv1d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=kernel_size_off,
                stride=stride_off,
                padding=padding_off,
                dilation=dilation_off,
                groups=groups * deformable_groups_per_groups,
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

        if self.offset_field is not None:
            nn.init.zeros_(self.offset_field.wieght)
            if self.offset_field.bias is not None:
                nn.init.zeros_(self.offset_field.bias)

        nn.init.zeros_(self.attn_mask.weight)
        if self.attn_mask.bias is not None:
            nn.init.ones_(self.attn_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset_field is None:
            offset_field = None
        else:
            offset_field = self.offset_field(x)

        attn_mask = self.attn_mask(x)

        return torch.ops.custom_op.deform_conv1d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups_per_groups,
            self.offset_scale,
            self.fix_center,
            self.bias
        )


class DeformConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups_per_groups: int = 1,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 7,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 3,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.fix_center = fix_center

        kernel_sizes = multiply_integers(self.kernel_size)

        if fix_center:
            assert kernel_sizes % 2 != 0

        if fix_center and kernel_sizes == 1:
            self.offset_field = None
        else:
            self.offset_field = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size_off,
                    stride=stride_off,
                    padding=padding_off,
                    dilation=dilation_off,
                    groups=in_channels
                ),
                group_norm(in_channels, 1),
                nn.Conv2d(
                    in_channels,
                    groups * deformable_groups_per_groups * (kernel_sizes - fix_center) * 2,
                    kernel_size=1,
                )
            )

        self.attn_mask = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size_off,
                    stride=stride_off,
                    padding=padding_off,
                    dilation=dilation_off,
                    groups=in_channels
                ),
                group_norm(in_channels, 1),
                nn.Conv2d(
                    in_channels,
                    groups * deformable_groups_per_groups * kernel_sizes,
                    kernel_size=1,
                )
            )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset_field is None:
            offset_field = None
        else:
            offset_field = self.offset_field(x)

        attn_mask = self.attn_mask(x)

        return torch.ops.custom_op.deform_conv2d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups_per_groups,
            self.offset_scale,
            self.fix_center,
            self.bias
        )


class DeformConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups_per_groups: int = 1,
                 offset_scale: float = 1.0,
                 fix_center: bool = False,
                 bias: bool = True,
                 kernel_size_off: Union[int, List[int], Tuple[int]] = 7,
                 stride_off: Union[int, List[int], Tuple[int]] = 1,
                 padding_off: Union[int, List[int], Tuple[int]] = 3,
                 dilation_off: Union[int, List[int], Tuple[int]] = 1,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.fix_center = fix_center

        kernel_sizes = multiply_integers(self.kernel_size)

        if fix_center:
            assert kernel_sizes % 2 != 0

        if fix_center and kernel_sizes == 1:
            self.offset_field = None
        else:
            self.offset_field = nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * (kernel_sizes - fix_center) * 3,
                kernel_size=kernel_size_off,
                stride=stride_off,
                padding=padding_off,
                dilation=dilation_off,
                groups=groups * deformable_groups_per_groups,
            )

        self.attn_mask = nn.Conv3d(
            in_channels,
            groups * deformable_groups_per_groups * kernel_sizes,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups * deformable_groups_per_groups,
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        nn.init.normal_(self.weight, mean=mean, std=std)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=mean, std=std)

        if self.offset_field is not None:
            nn.init.zeros_(self.offset_field.weight)
            if self.offset_field.bias is not None:
                nn.init.zeros_(self.offset_field.bias)

        nn.init.zeros_(self.attn_mask.weight)
        if self.attn_mask.bias is not None:
            nn.init.ones_(self.attn_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.offset_field is None:
            offset_field = None
        else:
            offset_field = self.offset_field(x)

        attn_mask = self.attn_mask(x)

        return torch.ops.custom_op.deform_conv3d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups_per_groups,
            self.offset_scale,
            self.fix_center,
            self.bias
        )


def deform_conv_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return DeformConv1d(*args, **kwargs)
    elif dim == 2:
        return DeformConv2d(*args, **kwargs)
    elif dim == 3:
        return DeformConv3d(*args, **kwargs)
    else:
        raise NotImplementedError(f'{dim} is not supported.')
