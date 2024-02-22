import torch
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import resize
from torch.nn import functional as F
import lightning as L

from models.losses.MaskedWingLoss import MaskedWingLoss
from models.baselines.HeatmapBasedLandmarkDetection \
    import HeatmapBasedLandmarkDetection
from models.baselines.hourglass.hourglass import HourglassNet


class KimLandmarkDetection(L.LightningModule, HeatmapBasedLandmarkDetection):
    def __init__(
        self,
        resize_to: tuple[int, int] = (256, 256),
        num_points: int = 44,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
        original_image_size: tuple[int, int] = (1840, 1360),
        *args,
        **kwargs
    ):
        super().__init__()

        self.patch_size = torch.tensor(resize_to)
        self.resize_to = torch.tensor(resize_to)
        self.original_image_size = original_image_size
        self.patch_resize_to = self._get_patch_resize_to()
        self.mm_loss = MaskedWingLoss(
            original_image_size=original_image_size,
            resize_to=resize_to
        )
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

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resized = resize(x, self.resize_to)  # batch_size, 1, 256, 256

        batch_size, channels, height, width = resized.shape

        global_heatmaps = self.global_module(
            resized
        )  # batch_size, 44, 256, 256
        point_predictions = self._get_highest_points(
            global_heatmaps
        )  # batch_size, 44, 2

        regions_of_interest = self._extract_patches(
            x, point_predictions
        )  # batch_size, 44, 256, 256

        local_heatmaps = self.local_module(
            regions_of_interest.view(
                batch_size * self.num_points,
                channels,
                height,
                width
            )
        ).view(
            batch_size,
            self.num_points,
            height,
            width
        )

        local_heatmaps = self._paste_heatmaps(
            global_heatmaps,
            local_heatmaps,
            point_predictions
        )

        refined_point_predictions = self._get_highest_points(
            local_heatmaps
        )  # batch_size, 44, 2

        return global_heatmaps, local_heatmaps, refined_point_predictions

    def step(self, batch, batch_idx):
        images, targets = batch

        global_heatmaps, local_heatmaps, predictions = self(images)

        target_heatmaps = self._get_heatmaps(
            targets
        )  # batch_size, 44, 256, 256

        loss = F.binary_cross_entropy_with_logits(
            global_heatmaps, target_heatmaps
        ) + F.binary_cross_entropy_with_logits(
            local_heatmaps, target_heatmaps
        )

        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx)

        self.log('train_loss', loss)

        return loss

    def validation_test_step(self, batch, batch_idx):
        loss, predictions = self.step(batch, batch_idx)
        targets = batch[1]

        _, unreduced_mm_error = self.mm_loss(
            predictions,
            targets,
            with_mm_error=True
        )

        return loss, unreduced_mm_error

    def validation_step(self, batch, batch_idx):
        loss, unreduced_mm_error = self.validation_test_step(batch, batch_idx)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mm_error', unreduced_mm_error.mean(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, unreduced_mm_error = self.validation_test_step(batch, batch_idx)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', unreduced_mm_error[id].mean())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', unreduced_mm_error.mean(), prog_bar=True)

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
