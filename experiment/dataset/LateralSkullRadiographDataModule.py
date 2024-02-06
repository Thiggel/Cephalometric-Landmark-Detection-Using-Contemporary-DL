import lightling as L
import torch
from torch.utils.data import random_split, DataLoader

import LateralSkullRadiographDataset


class LateralSkullRadiographDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str = '../../dataset/',
        csv_file: str = 'all_images_same_points.csv',
        transform: L.transforms.Compose = None,
        splits: tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size = 32,
    ):
        super().__init__()

        self.dataset = LateralSkullRadiographDataset(
            root_dir=root_dir,
            csv_file=csv_file,
            transform=transform
        )

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            torch.tensor(splits) * len(self.dataset)
        )

        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )
