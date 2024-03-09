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
        self,
        points: torch.Tensor,
        gaussian_sd: float = 1
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
            torch.arange(
                self.resized_image_size[0],
                device=self.device
            ),
            torch.arange(
                self.resized_image_size[1],
                device=self.device
            ),
        )

        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        x, y = points.split(1, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        heatmaps = torch.exp(
            -0.5 * ((y_grid - y) ** 2 + (x_grid - x) ** 2) / (gaussian_sd ** 2)
        )

        return heatmaps, mask.unsqueeze(-1).unsqueeze(-1)
