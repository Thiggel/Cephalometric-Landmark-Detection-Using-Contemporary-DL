import torch
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms.functional import resize
from torch.nn import functional as F
import lightning as L

from models.losses.MaskedWingLoss import MaskedWingLoss


class HourglassBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class HourglassNet(nn.Module):
    def __init__(self, num_blocks, num_channels):
        super(HourglassNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels

        self.down_blocks = nn.ModuleList([
            HourglassBlock(3, num_channels)
            for _ in range(num_blocks)
        ])

        self.up_blocks = nn.ModuleList([
            HourglassBlock(num_channels, num_channels)
            for _ in range(num_blocks)
        ])

        self.skip_connections = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, kernel_size=1)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        features = []

        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            features.append(x)
            x = nn.MaxPool2d(2, 2)(x)

        for i in range(self.num_blocks):
            x = self.up_blocks[i](x)
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            skip_connection = self.skip_connections[i](
                features[self.num_blocks - i - 1]
            )
            x = x + skip_connection

        return x


class KimLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        resize_to: tuple[int, int] = (256, 256),
        num_points: int = 44,
        num_hourglass_modules: int = 4,
        num_blocks_per_hourglass: int = 4,
        original_image_size: tuple[int, int] = (1360, 1840)
    ):
        super().__init__()

        self.resize_to = resize_to
        self.patch_resize_to = self._get_patch_resize_to()
        self.original_image_size = original_image_size
        self.mm_loss = MaskedWingLoss(
            original_image_size=original_image_size,
            resize_to=resize_to
        )
        self.num_points = num_points

        self.global_module = nn.Sequential(*[
            HourglassNet(num_blocks_per_hourglass, num_points)
            for _ in range(num_hourglass_modules)
        ])

        self.local_module = nn.Sequential(*[
            HourglassNet(num_blocks_per_hourglass, 1)
            for _ in range(num_hourglass_modules)
        ])

    def _get_patch_resize_to(self):
        resize_factor = self.resize_to[0] / self.original_image_size[0]
        patch_resize_to = (
            int(resize_factor * self.resize_to[0]),
            int(resize_factor * self.resize_to[1])
        )

        return patch_resize_to

    def _paste_heatmaps(
        self,
        global_heatmaps: torch.Tensor,
        local_heatmaps: torch.Tensor,
        point_predictions: torch.Tensor
    ) -> torch.Tensor:
        # each global heatmap has a shape of (44, 256, 256)
        # it contains one heatmap for each point

        # each local heatmap was created from patch of the original
        # unresized image, so a patch of 256x256 pixels was cut out
        # around the respective point prediction at from there on
        # a refined prediction was created.
        # The local heatmaps tensor thus has a shape of (44, 256, 256)
        # but each of the 44 heatmaps refers to a patch of the original
        # image.
        # This function pastes the local heatmaps back into the global
        # heatmaps at the respective position of the refined point
        # specifically, it first resizes the local heatmaps to be in the
        # same aspect ratio as the global heatmaps.
        # it then takes the point prediction from the point_predictions
        # tensor, and pastes the resized patch for each point into the
        # global heatmap at the respective position of the refined
        # point prediction.
        resized_local_heatmaps = resize(
            local_heatmaps,
            self.patch_resize_to,
        )

        for i in range(self.num_points):
            x, y = point_predictions[i].round().int()
            global_heatmaps[
                i,
                :,
                y:y + self.resize_to[1],
                x:x + self.resize_to[0]
            ] = resized_local_heatmaps[i]

        return global_heatmaps

    def forward(self, x):
        resized = resize(x, self.resize_to)  # batch_size, 1, 256, 256

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
            regions_of_interest
        )  # batch_size, 44, 256, 256
        local_heatmaps = self._paste_heatmaps(
            global_heatmaps,
            local_heatmaps,
            point_predictions
        )

        refined_point_predictions = self._refine_point_predictions(
            point_predictions, local_heatmaps
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

        mm_error = self.mm_loss.magnitude_to_mm(
            predictions,
            targets
        )

        return loss, mm_error

    def validation_step(self, batch, batch_idx):
        loss, mm_error = self.validation_test_step(batch, batch_idx)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mm_error', mm_error.mean(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, mm_error = self.validation_test_step(batch, batch_idx)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error.mean(), prog_bar=True)

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
