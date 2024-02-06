import lightning as L
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler


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

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        pass

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        pass

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        pass

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience
        )

        return [optimizer], [scheduler]
