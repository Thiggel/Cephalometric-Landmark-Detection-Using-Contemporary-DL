import lightning as L
from torch.utils.data import random_split, DataLoader
from typing import Callable

from dataset.LateralSkullRadiographDataset import LateralSkullRadiographDataset


class LateralSkullRadiographDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        resized_image_size: tuple[int, int],
        transform: Callable = None,
        splits: tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        flip_augmentations: bool = True,
    ):
        super().__init__()

        self.dataset = LateralSkullRadiographDataset(
            root_dir=root_dir,
            csv_file=csv_file,
            transform=transform,
            resized_image_size=resized_image_size,
            flip_augmentations=flip_augmentations,
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            self._get_splits(splits)
        )

        self.batch_size = batch_size

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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )
