import torch
from torch import nn
import torch.nn.functional as F
import gc


class HourglassBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class HourglassNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int
    ):
        super(HourglassNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_channels = out_channels

        self.down_blocks = nn.ModuleList([
            HourglassBlock(
                in_channels if block_idx == 0 else out_channels,
                out_channels
            )
            for block_idx in range(num_blocks)
        ])

        self.up_blocks = nn.ModuleList([
            HourglassBlock(out_channels, out_channels)
            for _ in range(num_blocks)
        ])

        self.skip_connections = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            for _ in range(num_blocks)
        ])

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = []

        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            features.append(x)
            x = self.max_pool(x)

        for i in range(self.num_blocks):
            x = (
                F.interpolate(
                    self.up_blocks[i](x),
                    scale_factor=2
                )
                +
                self.skip_connections[i](
                    features[self.num_blocks - i - 1]
                )
            )

        return x
