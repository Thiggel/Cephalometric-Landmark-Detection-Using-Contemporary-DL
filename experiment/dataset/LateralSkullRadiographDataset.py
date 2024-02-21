import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from PIL import Image
import ast
from tqdm import tqdm
from albumentations.augmentations.transforms import \
        GaussNoise

from utils.CanExtractPatches import CanExtractPatches


class LateralSkullRadiographDataset(Dataset, CanExtractPatches):
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        crop: bool = False,
        resize_to: tuple[int, int] = (450, 450),
        use_heatmaps: bool = False,
        transform: transforms.Compose = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5
            ),
            GaussNoise(var_limit=0.2, mean=0, p=0.5),
        ]),
        flip_augmentations: bool = True,
        original_image_size: tuple[int, int] = (1840, 1360),
        patch_size: tuple[int, int] = (96, 96),
    ):
        self.data_frame = pd.read_csv(
            os.path.join(root_dir, csv_file),
        )
        self.root_dir = root_dir
        self.crop = crop
        self.resize = transforms.Resize(resize_to)
        self.to_tensor = transforms.ToTensor()
        self.resize_to = resize_to
        self.use_heatmaps = use_heatmaps
        self.transform = transform
        self.flip_augmentations = flip_augmentations

        self.original_image_size = original_image_size
        self.patch_size = patch_size

        print('Loading dataset into memory...')
        (
            self.images,
            self.points,
            self.point_ids,
            self.heatmaps_full,
            self.heatmaps_patch
        ) = self._load_data()

        print('Done!')

    def _parse_dimensions(self, x: str) -> tuple[int, int]:
        try:
            return eval(x)
        except Exception:
            return None

    def _crop_image(self, image: torch.Tensor) -> torch.Tensor:
        original_height, original_width = image.shape[-2:]
        new_size = min(original_height, original_width)

        return crop(
            image,
            top=original_height - new_size,
            left=original_width - new_size,
            height=new_size,
            width=new_size,
        )

    def _load_image(self, index: int, resize=True) -> torch.Tensor:
        img_name = os.path.join(
            self.root_dir,
            f"images/{self.data_frame.iloc[index]['document']}.png"
        )

        image = Image.open(img_name).convert('L')

        image = self.to_tensor(image)

        if self.crop:
            image = self._crop_image(image)

        if resize:
            image = self.resize(image)

        return image

    @property
    def _saved_images_path(self) -> str:
        return os.path.join(self.root_dir, f'images_{self.resize_to}.pt')

    @property
    def _saved_points_path(self) -> str:
        return os.path.join(self.root_dir, f'points_{self.resize_to}.pt')

    @property
    def _saved_heatmaps_full_path(self) -> str:
        return os.path.join(self.root_dir, f'heatmaps_full_{self.resize_to}.pt')

    @property
    def _saved_heatmaps_patch_path(self) -> str:
        return os.path.join(self.root_dir, f'heatmaps_patch_{self.resize_to}.pt')

    def _load_dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        points = []
        heatmaps_full = []
        heatmaps_patch = []

        for index in tqdm(range(len(self.data_frame))):
            image = self._load_image(index)
            image_points = self._load_points(index)

            images.append(image)
            points.append(image_points)

            if self.use_heatmaps:
                heatmap_full, heatmap_patch = self._load_heatmaps(
                    image,
                    image_points,
                )

                heatmaps_full.append(heatmap_full)
                heatmaps_patch.append(heatmap_patch)

        return (
            torch.stack(images),
            torch.stack(points),
            torch.stack(heatmaps_full) if self.use_heatmaps else None,
            torch.stack(heatmaps_patch) if self.use_heatmaps else None,
        )

    def _cormalize_btw_0_and_1(self, images: torch.Tensor) -> torch.Tensor:
        return (images - images.min()) / (images.max() - images.min())

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        normalize = transforms.Normalize(
            mean=images.mean(),
            std=images.std(),
        )

        return normalize(images)

    def _generate_heatmap(
        self,
        target_points: torch.Tensor,
        images: torch.Tensor,
        radius_around_point: int = 5,
        gaussian_sigma: float = 2.0,
    ) -> torch.Tensor:
        _, height, width = images.shape
        num_points, _ = target_points.size()

        y = torch.arange(0, height).float()
        x = torch.arange(0, width).float()
        yy, xx = torch.meshgrid(y, x)

        heatmap = torch.zeros(height, width)
        for point_idx in range(num_points):
            x_center, y_center = target_points[point_idx]

            if (
                x_center <= 0 or x_center >= width or
                y_center <= 0 or y_center >= height
            ):
                continue

            heatmap += torch.exp(
                -0.5 * (
                    (xx - x_center) ** 2 +
                    (yy - y_center) ** 2
                ) / gaussian_sigma ** 2
            )

        return heatmap

    def _load_heatmaps(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_heatmap_full = self._generate_heatmap(
            points,
            image
        )

        target_heatmaps_patch = self._extract_patches(
            target_heatmap_full.unsqueeze(0),
            points.unsqueeze(0),
        ).squeeze(0)

        return target_heatmap_full.unsqueeze(0), target_heatmaps_patch

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_ids = self._load_point_ids()

        if os.path.exists(self._saved_images_path) \
                and os.path.exists(self._saved_points_path) \
                and not self.use_heatmaps or (
                    os.path.exists(self._saved_heatmaps_full_path)
                    and os.path.exists(self._saved_heatmaps_patch_path)
                ):

            images = torch.load(self._saved_images_path)
            points = torch.load(self._saved_points_path)

            heatmaps_full = torch.load(self._saved_heatmaps_full_path) \
                if self.use_heatmaps else None
            heatmaps_patch = torch.load(self._saved_heatmaps_patch_path) \
                if self.use_heatmaps else None

            return images, points, point_ids, heatmaps_full, heatmaps_patch

        images, points, heatmaps_full, heatmaps_patch = self._load_dataset()

        images = self._normalize(images)

        self._save_to_pickle(images, points, heatmaps_full, heatmaps_patch)

        return images, points, point_ids, heatmaps_full, heatmaps_patch

    def _load_point_ids(self) -> list[str]:
        points_str = self.data_frame.iloc[0]['points']
        points_dict = ast.literal_eval(points_str)

        ids = [key for key in points_dict]

        return ids

    def _resize_point(self, point: dict[str, float]) -> dict[str, float]:
        x_ratio = self.resize_to[1] / self.original_image_size[1]
        y_ratio = self.resize_to[0] / self.original_image_size[0]

        return [
            point['x'] * x_ratio,
            point['y'] * y_ratio,
        ]

    def _load_points(self, index: int, resize=True) -> list[torch.Tensor]:
        points_str = self.data_frame.iloc[index]['points']
        points_dict = ast.literal_eval(points_str)

        points = [
            self._resize_point(points_dict[key])
            if resize
            else [points_dict[key]['x'], points_dict[key]['y']]
            for key in points_dict
        ]

        return torch.Tensor(points)

    def _save_to_pickle(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        heatmaps_full: torch.Tensor,
        heatmaps_patch: torch.Tensor,
    ):
        torch.save(images, self._saved_images_path)
        torch.save(points, self._saved_points_path)

        if self.use_heatmaps:
            torch.save(heatmaps_full, self._saved_heatmaps_full_path)
            torch.save(heatmaps_patch, self._saved_heatmaps_patch_path)

    def __len__(self) -> int:
        return len(self.data_frame)

    def _flip_horizontally(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        heatmap_full: torch.Tensor,
        heatmaps_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.flip(image, [-1])

        flipped_points = points.clone()
        flipped_points[..., 0] = image.shape[-1] - flipped_points[..., 0]

        if self.use_heatmaps:
            heatmap_full = torch.flip(heatmap_full, [-1])
            heatmaps_patch = torch.flip(heatmaps_patch, [-1])

        return image, flipped_points, heatmap_full, heatmaps_patch

    def _flip_vertically(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        heatmap_full: torch.Tensor,
        heatmaps_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.flip(image, [1])

        flipped_points = points.clone()
        flipped_points[..., 1] = image.shape[-2] - flipped_points[..., 1]

        if self.use_heatmaps:
            heatmap_full = torch.flip(heatmap_full, [1])
            heatmaps_patch = torch.flip(heatmaps_patch, [2])

        return image, flipped_points, heatmap_full, heatmaps_patch

    def _flip_image(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        heatmap_full: torch.Tensor,
        heatmaps_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.rand(1) > 0.5:
            (
                image,
                points,
                heatmap_full,
                heatmaps_patch
            ) = self._flip_horizontally(
                image,
                points,
                heatmap_full,
                heatmaps_patch
            )

        if torch.rand(1) > 0.5:
            (
                image,
                points,
                heatmap_full,
                heatmaps_patch
            ) = self._flip_vertically(
                image,
                points,
                heatmap_full,
                heatmaps_patch
            )

        return image, points, heatmap_full, heatmaps_patch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        points = self.points[idx]
        heatmap_full = self.heatmaps_full[idx] \
            if self.use_heatmaps else torch.tensor([])
        heatmaps_patch = self.heatmaps_patch[idx] \
            if self.use_heatmaps else torch.tensor([])

        if self.transform is not None:
            image = self.transform(image)

        if self.flip_augmentations:
            image, points, heatmap_full, heatmaps_patch = self._flip_image(
                image,
                points,
                heatmap_full,
                heatmaps_patch
            )

        return image, points, heatmap_full, heatmaps_patch
