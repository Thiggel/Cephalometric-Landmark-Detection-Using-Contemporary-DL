import torch
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import lightning as L

from models.losses.MaskedWingLoss import MaskedWingLoss
from models.baselines.HeatmapBasedLandmarkDetection \
    import HeatmapBasedLandmarkDetection
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



class KimLandmarkDetection(L.LightningModule, HeatmapBasedLandmarkDetection):
    def __init__(
        self,
        resized_image_size: tuple[int, int] = (448, 448),
        resized_point_reference_frame_size: tuple[int, int] = (256, 256),
        num_points: int = 44,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
        original_image_size: tuple[int, int] = (1840, 1360),
        patch_size: tuple[int, int] = (256, 256),
        only_global_detection: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.resized_image_size = resized_image_size
        self.resized_point_reference_frame_size = resized_point_reference_frame_size
        self.original_image_size = original_image_size
        self.resized_patch_size = self._get_resized_patch_size()
        self.only_global_detection = only_global_detection
        self.num_points = num_points
        self.use_offset_maps = False

        self.global_module = nn.Sequential(*[
            HourglassNet(
                num_blocks_per_hourglass,
                in_channels=num_points if block_idx > 0 else 1,
                out_channels=num_points
            )
            for block_idx in range(num_hourglass_modules)
        ])

        self.local_module = nn.Sequential(*[
            HourglassNet(
                num_blocks_per_hourglass,
                in_channels=1,
                out_channels=1
            )
            for block_idx in range(num_hourglass_modules)
        ])

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
