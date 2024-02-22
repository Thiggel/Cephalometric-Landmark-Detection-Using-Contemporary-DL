import torch
from torch import nn


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module (https://arxiv.org/abs/1606.00915)
    """

    def __init__(self, in_channels: int, out_channels: int, dilations: int):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d)
            for d_idx, d in enumerate(dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum([block(x) for block in self.aspp_blocks])
