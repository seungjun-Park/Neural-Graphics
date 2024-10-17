from modules.ops.modules.deform_conv import DeformConv2d
import torch
import torch.nn as nn


device = 'cuda'

layer = nn.Sequential(
    DeformConv2d(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        padding=1,
    ),
    nn.GroupNorm(1, 4),
    nn.SiLU(),
).to(device)

inp = torch.randn(2, 2, 64, 64).to(device)

with torch.no_grad():
    output = layer(inp)

print(output.requires_grad)

