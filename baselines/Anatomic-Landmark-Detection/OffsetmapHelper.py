import torch


class OffsetmapHelper:
    def __init__(
        self,
        resized_image_size: tuple[int, int],
        offset_map_radius: float,
    ):
        self.resized_image_size = resized_image_size
        self.offset_map_radius = offset_map_radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_offset_maps(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = targets.shape

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.resized_image_size[0], device=self.device),
            torch.arange(self.resized_image_size[1], device=self.device),
        )

        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        x, y = targets.split(1, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        x_offset_maps = (x - x_grid).unsqueeze(2)
        y_offset_maps = (y - y_grid).unsqueeze(2)

        offset_maps = (
            torch.cat([x_offset_maps, y_offset_maps], dim=2) / self.offset_map_radius
        )

        return offset_maps
