import lightning as L
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler

from models.ViT import ViT
from models.ConvNextV2 import ConvNextV2


class CephalometricLandmarkDetector(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        point_ids: list[str],
        reduce_lr_patience: int = 25,
        model_size: str = 'tiny',
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_size = model_size
        self.reduce_lr_patience = reduce_lr_patience
        self.model = self._init_model(model_name)
        self.point_ids = point_ids

    def _init_model(self, model_name: str) -> nn.Module:
        model_types = {
            'ViT': lambda model_size: ViT(model_size),
            'ConvNextV2': lambda model_size: ConvNextV2(model_size),
        }

        return model_types[model_name](self.model_size)

    def forward(self, x):
        return self.model(x)

    def masked_l2_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        l2_loss = ((predictions - targets) ** 2).sum(-1).sqrt()

        mask = (targets > 0).prod(-1)
        masked_l2_loss = l2_loss * mask

        return masked_l2_loss

    def get_mm_error(
        self,
        non_reduced_loss: torch.Tensor,
    ) -> torch.Tensor:
        px_per_m = torch.tensor(7_756)
        m_to_mm = 1000

        return (
            non_reduced_loss
            / px_per_m.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            * m_to_mm
        ).squeeze().mean(0)

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        images, points = batch

        predictions = self.model(images)

        non_reduced_loss = self.masked_l2_loss(predictions, points)

        mm_error = self.get_mm_error(
            non_reduced_loss
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

        mm_error = mm_error.mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error = self.step(batch, with_mm_error=True)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean(), logger=False)

        mm_error = mm_error.mean()

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
