import lightning as L
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
import torch.nn.functional as F


class CephalometricLandmarkDetector(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        reduce_lr_patience: int = 25,
    ):
        super().__init__()

        self.model = model
        self.reduce_lr_patience = reduce_lr_patience

    def forward(self, x):
        return self.model(x)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        images, points = batch

        predictions = self.model(images)

        loss = F.l1_loss(predictions, points)

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss = self.step(batch)

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
        loss = self.step(batch)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss = self.step(batch)

        self.log('test_loss', loss, prog_bar=True)

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
