import lightning as L
import torch
from torch import nn
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR
from typing import Union
import matplotlib.pyplot as plt
from utils.clamp_points import clamp_points

from models.losses.MaskedWingLoss import MaskedWingLoss
from models.metrics.MeanRadialError import MeanRadialError


class DirectPointPredictionBasedLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        point_ids: list[str],
        original_image_size_mm: tuple[float, float],
        original_image_size: tuple[int, int],
        resized_image_size: int = (224, 224),
        reduce_lr_patience: int = 25,
        optimizer: str = 'sgd_momentum',
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.reduce_lr_patience = reduce_lr_patience
        self.model = model
        self.point_ids = point_ids
        self.optimizer_name = optimizer

        self.loss = MaskedWingLoss()
        self.mean_radial_error = MeanRadialError(
            resized_image_size=resized_image_size,
            original_image_size_mm=original_image_size_mm,
        )

    def forward(self, x):
        return self.model(x)

    def show_images(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        preds = self(images)

        preds = clamp_points(preds, images).cpu().numpy()
        targets = clamp_points(targets, images).cpu().numpy()

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        num_samples = images.shape[0]

        fig, axs = plt.subplots(1, num_samples, figsize=(20, 20 * num_samples))

        for i, (image, target, pred) in enumerate(zip(images, targets, preds)):
            axis = axs if num_samples == 1 else axs[i]
            axis.imshow(image, cmap='gray')
            axis.scatter(*zip(*target), color='red', s=20)
            axis.scatter(*zip(*pred), color='blue', s=20)
            axis.axis('off')

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        inputs, targets = batch

        predictions = self.model(inputs)

        loss = self.loss(
            predictions,
            targets,
        )

        unreduced_mm_error = self.mean_radial_error(
            predictions,
            targets,
        ) if with_mm_error else None

        return loss, unreduced_mm_error, predictions, targets

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _, _, _ = self.step(batch, with_mm_error=True)

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
            self.log(f'{point_id}_mm_error', mm_error.mean(dim=0)[id].mean())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error.mean(), prog_bar=True)

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

    def get_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
        optimizers = {
            'adam': lambda: Adam(self.parameters(), lr=0.001),
            'rmsprop': lambda: RMSprop(self.parameters(), lr=0.001),
            'sgd': lambda: SGD(self.parameters(), lr=0.001),
            'sgd_momentum': lambda: SGD(
                self.parameters(),
                lr=0.001,
                momentum=0.9
            ),
        }

        return optimizers[optimizer]()

    def configure_optimizers(self) -> dict:
        optimizer = self.get_optimizer(self.optimizer_name)

        datamodule = self.trainer.datamodule
        train_dataloader = datamodule.train_dataloader()

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
