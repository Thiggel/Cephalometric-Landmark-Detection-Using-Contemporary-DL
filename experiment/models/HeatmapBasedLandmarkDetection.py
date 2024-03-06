import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR

from utils.HeatmapHelper import HeatmapHelper
from utils.OffsetmapHelper import OffsetmapHelper


class HeatmapBasedLandmarkDetection:
    def __init__(
        self,
        global_module: nn.Module,
        local_module: nn.Module,
        point_ids: list[str] = [],
        resized_image_size: tuple[int, int] = (448, 448),
        resized_point_reference_frame_size: tuple[int, int] = (256, 256),
        num_points: int = 44,
        original_image_size: tuple[int, int] = (1840, 1360),
        patch_size: tuple[int, int] = (256, 256),
        only_global_detection: bool = False,
        use_offset_maps: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.global_module = global_module
        self.local_module = local_module

        self.point_ids = point_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.resized_point_reference_frame_size = resized_point_reference_frame_size
        self.only_global_detection = only_global_detection
        self.num_points = num_points
        self.use_offset_maps = use_offset_maps

        self.heatmap_helper = HeatmapHelper(
            original_image_size,
            resized_image_size,
            resized_point_reference_frame_size,
            patch_size,
        )

        self.offsetmap_helper = OffsetmapHelper(
            resized_image_size,
            offset_map_radius=20,
        )

        self.mm_error = MaskedWingLoss(
            original_image_size=original_image_size,
            resized_image_size=resized_image_size
        )

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _ensure_correct_image_size(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.shape[-2:] != self.resized_point_reference_frame_size:
            images = F.interpolate(
                x,
                size=self.resized_point_reference_frame_size
            )

        return images

    def forward_with_heatmaps(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        images = self._ensure_correct_image_size(images)
    
        batch_size, channels, height, width = images.shape

        output = self.global_module(
            images
        )

        offset_maps = output[:, :2 * self.num_points].view(
            batch_size, self.num_points, 2, height, width
        ) if self.use_offset_maps else None

        global_heatmaps = output[:, 2 * self.num_points:] \
            if self.use_offset_maps else output

        point_predictions = self.heatmap_helper.get_highest_points(
            global_heatmaps
        )

        if self.only_global_detection:
            return (
                global_heatmaps,
                global_heatmaps,
                point_predictions,
                offset_maps
            )

        regions_of_interest = self.heatmap_helper.extract_patches(
            images, point_predictions
        )

        patch_height, patch_width = self.patch_size

        local_heatmaps = self.local_module(
            regions_of_interest.view(
                batch_size * self.num_points,
                channels,
                patch_height,
                patch_width
            )
        ).view(
            batch_size,
            self.num_points,
            patch_height,
            patch_width
        )

        pasted_local_heatmaps = self.heatmap_helper.paste_heatmaps(
            global_heatmaps,
            local_heatmaps,
            point_predictions,
        )

        refined_point_predictions = self.heatmap_helper.get_highest_points(
            pasted_local_heatmaps
        )

        return (
            global_heatmaps,
            local_heatmaps,
            pasted_local_heatmaps,
            offset_maps,
            point_predictions,
            refined_point_predictions,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, _, _, refined_predictions = self.forward_with_heatmaps(images)

        return refined_predictions

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = batch

        (
            global_heatmaps,
            local_heatmaps,
            pasted_local_heatmaps,
            offset_maps,
            predictions,
            refined_predictions
        ) = self.forward_with_heatmaps(images)

        target_global_heatmaps, mask = self._create_heatmaps(targets)
        target_local_heatmaps, _ = self._create_heatmaps(
            predictions
        )
        target_offset_maps = self._create_offset_maps(targets) \
            if self.use_offset_maps else None

        offset_loss = self.loss(
            offset_maps,
            target_offset_maps
        ).mean(2) if self.use_offset_maps else 0

        loss = self.loss(
            global_heatmaps,
            target_global_heatmaps,
        ) + self.loss(
            local_heatmaps,
            target_local_heatmaps,
        ) + offset_loss

        masked_loss = loss * mask

        return (
            masked_loss.mean(),
            refined_predictions,
            global_heatmaps,
            pasted_local_heatmaps,
            target_global_heatmaps
        )

    def validation_test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, predictions, _, _, _ = self.step(batch)
        targets = batch[1]

        _, unreduced_mm_error = self.mm_error(
            predictions,
            targets,
            with_mm_error=True
        )

        return loss, unreduced_mm_error, predictions, targets

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss, _, _,  _, _ = self.step(batch)

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

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
