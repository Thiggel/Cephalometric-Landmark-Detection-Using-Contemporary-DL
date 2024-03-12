import torch
import torch.nn.functional as F


class HeatmapHelper:
    def __init__(
        self,
        resized_image_size: tuple[int, int],
    ):
        self.resized_image_size = resized_image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_heatmaps(
        self, points: torch.Tensor, radius: float = 1
    ) -> torch.Tensor:
        """
        Create heatmaps for target points.
        The resulting tensor's shape will be
        (batch_size, num_points, image_height, image_width).
        A mask is returned alongside for points where
        one coordinate is negative. These can then be filtered out
        of the loss.
        """
        batch_size, num_points, _ = points.shape

        mask = (points[..., 0] >= 0) & (points[..., 1] >= 0)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.resized_image_size[0], device=self.device),
            torch.arange(self.resized_image_size[1], device=self.device),
        )

        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        y, x = points.split(1, dim=-1)
        y = y.unsqueeze(-2)
        x = x.unsqueeze(-1)

        distance = ((x_grid - x) ** 2 + (y_grid - y) ** 2).sqrt()

        heatmaps = torch.zeros(
            (batch_size, num_points, *self.resized_image_size),
            device=self.device
        )

        heatmaps[distance <= radius] = 1

        return heatmaps, mask.unsqueeze(-1).unsqueeze(-1)
