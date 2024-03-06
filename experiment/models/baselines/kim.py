from torch import nn

from models.baselines.hourglass.hourglass import HourglassNet


class KimGlobalModule(nn.Sequential):
    def __init__(
        self,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
        num_points: int = 44,
    ):
        super().__init__(*[
            HourglassNet(
                num_blocks_per_hourglass,
                in_channels=num_points if block_idx > 0 else 1,
                out_channels=num_points
            )
            for block_idx in range(num_hourglass_modules)
        ])


class KimLocalModule(nn.Sequential):
    def __init__(
        self,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
    ):
        super().__init__(*[
            HourglassNet(
                num_blocks_per_hourglass,
                in_channels=1,
                out_channels=1
            )
            for block_idx in range(num_hourglass_modules)
        ])
