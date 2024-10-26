import torch
import custom_op
import torch.nn.functional as F
from modules.blocks.deform_conv import deform_conv_nd


device = 'cuda'
dim = 2

layer = deform_conv_nd(
    dim=dim,
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    padding=1,
    stride=1,
    dilation=1,
    groups=8,
    offset_field_channels_per_groups=2,
    bias=True,
    modulation_type='none',
    kernel_size_off=3,
    padding_off=2,
    dilation_off=2
).to(device)

inp = torch.randn(2, 32, 64, 64).to(device)
weight = torch.randn(64, 32 // 8, 3, 3).to(device)
bias = torch.randn(64).to(device)
offset_field = torch.randn(2, 2, 16 * 9, 64, 64).to(device)
attn_mask = torch.randn(2, 1, 16 * 9, 64, 64).to(device)

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    # output = torch.ops.custom_op.deform_conv2d(
    #     inp,
    #     weight,
    #     offset_field,
    #     attn_mask,
    #     (3, 3),
    #     (1, 1),
    #     (1, 1),
    #     (1, 1),
    #     8,
    #     2,
    #     bias,
    # )
    output = layer(inp)

loss = F.l1_loss(torch.zeros_like(output), output).mean()
loss.backward()

print(output)
