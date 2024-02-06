import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import ast
from tqdm import tqdm


class LateralSkullRadiographDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        base_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        transform: transforms.Compose = None
    ):
        self.data_frame = pd.read_csv(
            os.path.join(root_dir, csv_file),
        )
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.transform = transform

        print('Loading dataset into memory...')
        self.images, self.points = self._load_data()
        print('Done!')

    def _load_image(self, index: int) -> torch.Tensor:
        img_name = os.path.join(
            self.root_dir,
            f"images/{self.data_frame.iloc[index]['document']}.png"
        )

        image = Image.open(img_name).convert('L')
        image = self.base_transform(image)

        return image

    @property
    def _saved_images_path(self) -> str:
        return os.path.join(self.root_dir, 'images.pt')

    @property
    def _saved_points_path(self) -> str:
        return os.path.join(self.root_dir, 'points.pt')

    def _load_dataset(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images = []
        points = []

        for index in tqdm(range(len(self.data_frame))):
            images.append(self._load_image(index))
            points.append(self._load_points(index))

        return torch.stack(images), torch.stack(points)

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if os.path.exists(self._saved_images_path) \
                and os.path.exists(self._saved_points_path):
            images = torch.load(self._saved_images_path)
            points = torch.load(self._saved_points_path)

            return images, points

        images, points = self._load_dataset()

        self._save_to_pickle(images, points)

        return images, points

    def _load_points(self, index: int) -> list[torch.Tensor]:
        points_str = self.data_frame.iloc[index]['points']
        points_dict = ast.literal_eval(points_str)

        points = [
            [points_dict[key]['x'], points_dict[key]['y']]
            for key in points_dict
        ]

        return torch.Tensor(points)

    def _save_to_pickle(self, images: torch.Tensor, points: torch.Tensor):
        torch.save(images, os.path.join(self.root_dir, 'images.pt'))
        torch.save(points, os.path.join(self.root_dir, 'points.pt'))

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        points = self.points[idx]

        if self.transform:
            image = self.transform(image)

        return image, points
