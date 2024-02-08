from datetime import date
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from utils.set_seed import set_seed
from loggers.ImagePredictionLogger import ImagePredictionLogger
from dataset.LateralSkullRadiographDataModule import LateralSkullRadiographDataModule
from models.CephalometricLandmarkDetector import CephalometricLandmarkDetector
from models.ViT import ViT


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dataset')
    parser.add_argument(
        '--csv_file', type=str, default='all_images_same_points.csv'
    )
    parser.add_argument('--splits', type=tuple, default=(0.8, 0.1, 0.1))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--early_stopping_patience', type=int, default=100)

    args = parser.parse_args()

    datamodule = LateralSkullRadiographDataModule(
        root_dir=args.root_dir,
        csv_file=args.csv_file,
        splits=args.splits,
        batch_size=args.batch_size
    )

    model = CephalometricLandmarkDetector(
        model=ViT()
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min',
    )

    tensorboard_logger = TensorBoardLogger(
        'logs/',
        name=model.model.__class__.__name__ + ' ' + date.today().isoformat(),
    )

    image_logger = ImagePredictionLogger(num_samples=5)

    trainer = L.Trainer(
        max_epochs=10_000,
        callbacks=[checkpoint_callback, early_stopping_callback, image_logger],
        enable_checkpointing=True,
        logger=tensorboard_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto'
    )

    trainer.fit(model=model, datamodule=datamodule)

    model = CephalometricLandmarkDetector.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    trainer.test(model=model, datamodule=datamodule)
