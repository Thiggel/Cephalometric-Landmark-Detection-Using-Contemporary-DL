import lightning as L
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
import torch.nn.functional as F
from typing import Callable

from models.ViT import ViT


class CephalometricLandmarkDetector(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        reduce_lr_patience: int = 25,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.image_size = image_size
        self.reduce_lr_patience = reduce_lr_patience
        self.model = self._init_model(model_name)

    @property
    def available_models(self) -> dict[str, Callable]:
        return {
            'ViT': lambda: ViT()
        }

    def _init_model(self, model_name: str) -> nn.Module:
        if model_name not in self.available_models:
            print('Invalid Model')
            exit()

        return self.available_models[model_name]()

    def forward(self, x):
        return self.model(x)

    def masked_l1_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        mask = targets > 0
        masked_predictions = predictions * mask
        masked_targets = targets * mask

        return F.l1_loss(masked_predictions, masked_targets, reduction='none')

    def get_mm_error(
        self,
        non_reduced_loss: torch.Tensor,
        image_dimensions: torch.Tensor
    ) -> torch.Tensor:
        print(non_reduced_loss * image_dimensions.unsqueeze(1) / self.image_size)
        print((non_reduced_loss * image_dimensions.unsqueeze(1) / self.image_size).mean())
        exit()
        return (
            non_reduced_loss * image_dimensions.unsqueeze(1)
            / self.image_size
        ).mean()

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        images, points, image_dimensions = batch

        predictions = self.model(images)

        non_reduced_loss = self.masked_l1_loss(predictions, points)

        mm_error = self.get_mm_error(
            non_reduced_loss,
            image_dimensions
        ) if with_mm_error else None

        loss = non_reduced_loss.mean()

        return loss, mm_error

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _ = self.step(batch)

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
        loss, mm_error = self.step(batch, with_mm_error=True)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error = self.step(batch, with_mm_error=True)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error, prog_bar=True)

        return loss

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
