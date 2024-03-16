from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from models.losses.HeatmapOffsetmapLoss import HeatmapOffsetmapLoss
from models.metrics.MeanRadialError import MeanRadialError
from utils.HeatmapHelper import HeatmapHelper


class HeatmapBasedLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        point_ids: list[str],
        original_image_size: tuple[float, float],
        original_image_size_mm: tuple[float, float],
        resized_image_size: int = (800, 640),
        batch_size: int = 1,
        output_size: int = 19,
        reduce_lr_patience: int = 25,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = model

        self.loss = HeatmapOffsetmapLoss(resized_image_size)

        self.mean_radial_error = MeanRadialError(
            resized_image_size=resized_image_size,
            original_image_size_mm=original_image_size_mm,
        )

        self.point_ids = point_ids

        self.reduce_lr_patience = reduce_lr_patience

        self.heatmap_helper = HeatmapHelper(
            original_image_size=original_image_size,
            resized_image_size=resized_image_size,
        )


    def forward_with_heatmaps(self, x):
        x = x.repeat(1, 3, 1, 1)
        output = self.model(x)
        return output

    def forward(self, x):
        output = self.forward_with_heatmaps(x)

        return self.get_points(output)

    def get_points(self, model_output: torch.Tensor):
        num_points = model_output.size(1) // 3
        return self.heatmap_helper.get_highest_points(
            model_output[:, :num_points]
        )

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        inputs, targets = batch

        predictions = self.forward_with_heatmaps(inputs)

        loss = self.loss(
            predictions,
            targets,
        )

        point_predictions = self.get_points(predictions)

        unreduced_mm_error = self.mean_radial_error(
            point_predictions,
            targets,
        ) if with_mm_error else None

        return loss, unreduced_mm_error, point_predictions, targets

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _, _, _ = self.step(batch)

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error, _, _ = self.step(batch, with_mm_error=True)

        mm_error = mm_error.mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        (
            loss,
            mm_error,
            predictions,
            targets
        ) = self.step(batch, with_mm_error=True)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean())

        mm_error = mm_error.mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error, prog_bar=True)

        self.log(
            'percent_under_1mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 1)
        )
        self.log(
            'percent_under_2mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 2)
        )
        self.log(
            'percent_under_3mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 3)
        )
        self.log(
            'percent_under_4mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 4)
        )

        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1.0
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    patience=self.reduce_lr_patience
                ),
                'monitor': 'val_loss'
            },
        }

