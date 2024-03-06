import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
from typing import Callable
import os

from dataset.LateralSkullRadiographDataset import LateralSkullRadiographDataset


class LateralSkullRadiographDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str = '../../dataset/',
        csv_file: str = 'all_images_same_points.csv',
        transform: Callable = None,
        splits: tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        crop: bool = False,
        resized_image_size: tuple[int, int] = (224, 224),
        resized_point_reference_frame_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.dataset = LateralSkullRadiographDataset(
            root_dir=root_dir,
            csv_file=csv_file,
            transform=transform,
            crop=crop,
            resized_image_size=resized_image_size,
            resized_point_reference_frame_size=resized_point_reference_frame_size
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            self._get_splits(splits)
        )

        self.batch_size = batch_size

        self.num_workers = 0 if torch.cuda.is_available() else 0

    def _get_splits(
        self,
        splits: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        size = len(self.dataset)

        train_size = int(splits[0] * size)
        val_size = int(splits[1] * size)
        test_size = size - train_size - val_size

        return train_size, val_size, test_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #persistent_workers=True
        )
