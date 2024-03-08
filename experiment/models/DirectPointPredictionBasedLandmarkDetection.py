import lightning as L
import torch
from torch import nn
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR

from models.losses.MaskedWingLoss import MaskedWingLoss


class DirectPointPredictionBasedLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        point_ids: list[str],
        reduce_lr_patience: int = 25,
        optimizer: str = 'adam',
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

    def forward(self, x):
        return self.model(x)

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        inputs, targets = batch

        predictions = self.model(inputs)

        loss, unreduced_mm_error = self.loss(
            predictions,
            targets,
            with_mm_error=with_mm_error,
        )

        return loss, unreduced_mm_error, predictions, targets

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
            self.loss.percent_under_n_mm(predictions, targets, 1)
        )
        self.log(
            'percent_under_2mm',
            self.loss.percent_under_n_mm(predictions, targets, 2)
        )
        self.log(
            'percent_under_3mm',
            self.loss.percent_under_n_mm(predictions, targets, 3)
        )
        self.log(
            'percent_under_4mm',
            self.loss.percent_under_n_mm(predictions, targets, 4)
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

        # Linear warmup scheduler
        def warmup_lr_lambda(current_step):
            if current_step <= self.warmup_epochs * len(train_dataloader):
                return current_step / (self.warmup_epochs * len(train_dataloader))
            else:
                return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

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
