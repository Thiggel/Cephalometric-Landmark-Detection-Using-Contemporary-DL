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


class KimLandmarkDetection(L.LightningModule, HeatmapBasedLandmarkDetection):
    def __init__(
        self,
        resize_to: tuple[int, int] = (448, 448),
        resize_points_to_aspect_ratio: tuple[int, int] = (256, 256),
        num_points: int = 44,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
        original_image_size: tuple[int, int] = (1840, 1360),
        patch_size: tuple[int, int] = (256, 256),
        *args,
        **kwargs
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.resize_to = resize_to
        self.resize_points_to_aspect_ratio = resize_points_to_aspect_ratio
        self.original_image_size = original_image_size
        self.patch_resize_to = self._get_patch_resize_to()
        self.num_points = num_points

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

        self.mm_error = MaskedWingLoss(
            original_image_size=original_image_size,
            resize_to=resize_to
        )
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward_with_heatmaps(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resized = F.interpolate(
            x,
            size=self.resize_points_to_aspect_ratio
        )  # batch_size, 1, 256, 256

        return self.forward_batch(resized, resized.shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, predictions = self.forward_with_heatmaps(x)

        return predictions

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.step(batch)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, unreduced_mm_error, _, _ = self.validation_test_step(batch)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mm_error', unreduced_mm_error.mean(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        (
            loss,
            unreduced_mm_error,
            predictions,
            targets
        ) = self.validation_test_step(batch)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', unreduced_mm_error[id].mean())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', unreduced_mm_error.mean(), prog_bar=True)

        self.log(
            'percent_under_1mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 1)
        )
        self.log(
            'percent_under_2mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 2)
        )
        self.log(
            'percent_under_3mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 3)
        )
        self.log(
            'percent_under_4mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 4)
        )

        return loss

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(), lr=2.5e-4)
        scheduler = ExponentialLR(optimizer, 0.9)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 4,
                'monitor': 'val_loss',
            }
        }
