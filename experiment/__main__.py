from datetime import date
import time
import argparse
import lightning as L
from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        DeviceStatsMonitor
    )
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.multiprocessing as mp

from utils.set_seed import set_seed
from loggers.ImagePredictionLogger import ImagePredictionLogger
from dataset.LateralSkullRadiographDataModule import \
    LateralSkullRadiographDataModule
from models.ModelTypes import ModelTypes

mp.set_start_method('spawn')


def get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dataset/novel')
    parser.add_argument(
        '--csv_file', type=str, default='all_images_37_points.csv'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        choices=ModelTypes.get_model_types()
    )
    parser.add_argument('--splits', type=tuple, default=(0.8, 0.1, 0.1))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--early_stopping_patience', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_only', action=argparse.BooleanOptionalAction)
    parser.add_argument('--flip_augmentations', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--max_hours_per_run', type=int, default=5)
    parser.add_argument('--logger', action='store_true')
    parser.add_argument('--no-logger', action='store_false', dest='logger')
    parser.set_defaults(logger=True)

    args = parser.parse_args()

    return args


def get_model_name(module: L.LightningModule) -> str:
    if not hasattr(module, 'model'):
        return module.__class__.__name__

    return module.model.__class__.__name__

def run(args: dict, seed: int = 42) -> dict:
    set_seed(seed)

    model_type = ModelTypes.get_model_type(args.model_name)

    datamodule = LateralSkullRadiographDataModule(
        root_dir=args.root_dir,
        csv_file=args.csv_file,
        splits=args.splits,
        batch_size=args.batch_size,
        resized_image_size=model_type.resized_image_size,
        flip_augmentations=args.flip_augmentations,
    )

    model_args = {
        'model_name': args.model_name,
        'point_ids': datamodule.dataset.point_ids,
        'output_size': datamodule.dataset.num_points,
        'original_image_size': datamodule.dataset.original_image_size,
        'original_image_size_mm': datamodule.dataset.original_image_size_mm,
        'resized_image_size': model_type.resized_image_size,
        'batch_size': args.batch_size,
    }

    model = model_type.initialize(**model_args)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename=args.model_name + ' ' + args.csv_file + '-{epoch}-{val_loss:.2f}',
        monitor='val_loss',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min',
    )

    tensorboard_logger = TensorBoardLogger(
        'logs/',
        name=args.model_name + ' ' + args.csv_file
    )

    image_logger = ImagePredictionLogger(
        num_samples=5,
        resized_image_size=model_type.resized_image_size,
        model_name=args.model_name,
        dataset_name=args.csv_file,
    )

    if args.checkpoint is not None:
        model.load_state_dict(
            torch.load(
                args.checkpoint,
                map_location=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            )['state_dict']
        )

    stats_monitor = DeviceStatsMonitor()

    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        image_logger,
        stats_monitor
    ]

    trainer_args = {
        'max_time': {'hours': args.max_hours_per_run},
        'max_epochs': 10_000,
        'callbacks': callbacks,
        'enable_checkpointing': True,
        'logger': tensorboard_logger if args.logger else None,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 'auto',
    }

    trainer = L.Trainer(
        **trainer_args
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)

        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)['state_dict']
        )

    return trainer.test(model=model, datamodule=datamodule)


def print_mean_std(results: list[dict]) -> None:
    tensor_data = torch.tensor([list(d.values()) for d in results])

    mean = tensor_data.mean(dim=0)
    std_dev = tensor_data.std(dim=0)

    for key, m, s in zip(results[0].keys(), mean, std_dev):
        print(f'{key} - Mean: {m.item()}, Std: {s.item()}')


if __name__ == '__main__':
    args = get_args()

    all_results = []

    for run_idx in range(args.num_runs):
        start_time = time.time()

        results = run(args, seed=run_idx)[0]

        end_time = time.time()
        seconds_to_hours = 3600
        training_time = (end_time - start_time) / seconds_to_hours
        results.update({'training_time': training_time})

        print(results)

        all_results.append(results)

    print_mean_std(all_results)
