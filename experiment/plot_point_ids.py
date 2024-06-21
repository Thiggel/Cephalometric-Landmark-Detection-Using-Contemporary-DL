import ast
from tqdm import tqdm
import os
import yaml
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt

class PlotImage:

    def __init__(self, root_dir, csv_file, resized_image_size):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.resized_image_size = resized_image_size
        self.to_tensor = ToTensor()
        self.resize = Resize(size=self.resized_image_size)

        self._get_metadata()
        self.data_frame = pd.read_csv(
            os.path.join(root_dir, csv_file),
            dtype={
                'document': str,
                'points': str,
            }
        )

        self.point_ids = self._load_point_ids()

        for point_id in self.point_ids:
            print(point_id)

        exit()

    def _get_metadata(self) -> dict:
        metadata_file = os.path.join(
            self.root_dir,
            'metadata.yaml'
        )

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as file:
                metadata = yaml.safe_load(file)

                self.original_image_size = (
                    metadata['image_height'],
                    metadata['image_width'],
                )

                self.original_image_size_mm = (
                    metadata['image_height_mm'],
                    metadata['image_width_mm'],
                )

    def _load_point_ids(self) -> list[str]:
        points_str = self.data_frame.iloc[0]['points']
        points_dict = ast.literal_eval(points_str)

        ids = [key for key in points_dict]

        return ids

    def _resize_point(self, point: dict[str, float]) -> dict[str, float]:
        x_ratio = self.resized_image_size[1] / self.original_image_size[1]
        y_ratio = self.resized_image_size[0] / self.original_image_size[0]

        resized = [
            point['x'] * x_ratio,
            point['y'] * y_ratio,
        ]

        if resized[0] > self.resized_image_size[1]:
            resized[0] = -1

        if resized[1] > self.resized_image_size[0]:
            resized[1] = -1

        return resized

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

    def _load_image(self, index: int, resize=True) -> torch.Tensor:
            img_name = os.path.join(
                self.root_dir,
                f"images/{self.data_frame.iloc[index]['document']}"
            )

            if not os.path.exists(img_name):
                if os.path.exists(img_name + '.jpg'):
                    img_name += '.jpg'

                if os.path.exists(img_name + '.png'):
                    img_name += '.png'

            image = Image.open(img_name).convert('RGB')

            image = self.to_tensor(image)

            if resize:
                image = self.resize(image)

            return image

    def plot_image(self):

        for index in tqdm(range(len(self.data_frame))):
            points = self._load_points(index)

            if torch.all(points > 0):
                image = self._load_image(index)
                break


        plt.imshow(image.permute(1, 2, 0))

        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c='r', s=2)
            # make the point number be in a little box
            plt.text(point[0], point[1], str(self.point_ids[i]), color='darkred', fontsize=4)

        plt.axis('off')

        plt.savefig('image_with_points.pdf', format='pdf', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    plot_image = PlotImage(
        root_dir='dataset/benchmark',
        csv_file='points.csv',
        resized_image_size=(3000, 3000),
    )

    plot_image.plot_image()
