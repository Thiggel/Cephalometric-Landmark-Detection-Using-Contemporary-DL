from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapOffsetmapLoss(nn.Module):
    def __init__(
        self,
        resized_image_size: tuple[int, int],
        heatmap_radius: int = 40,
        offsetmap_radius: int = 40,
        gaussian: bool = True,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.heatmap_radius = heatmap_radius
        self.offsetmap_radius = offsetmap_radius

        self.init_offset_and_heatmaps(
            resized_image_size,
            heatmap_radius,
            offsetmap_radius,
            gaussian
        )

    def init_offset_and_heatmaps(
        self,
        resized_image_size: tuple[int, int],
        heatmap_radius: int,
        offsetmap_radius: int,
        gaussian: bool,
    ):
        height, width = resized_image_size

        self.init_general_heatmap(height, width, heatmap_radius, gaussian)
        self.init_general_offsetmap_x(height, width, offsetmap_radius)
        self.init_general_offsetmap_y(height, width, offsetmap_radius)

    def init_general_offsetmap_x(
        self,
        height: int,
        width: int,
        offsetmap_radius: int
    ):
        self.general_offsetmap_x = torch.arange(
            height * 2, device=self.device, dtype=torch.float32
        ).view(-1, 1).expand(height * 2, width * 2)

        self.general_offsetmap_x = height - self.general_offsetmap_x
        self.general_offsetmap_x /= offsetmap_radius

    def init_general_offsetmap_y(
        self,
        height: int,
        width: int,
        offsetmap_radius: int
    ):
        self.general_offsetmap_y = width - torch.arange(
            width * 2, device=self.device, dtype=torch.float32
        ).view(1, -1).expand(height * 2, width * 2)

        self.general_offsetmap_y = width - self.general_offsetmap_y

        self.general_offsetmap_y /= offsetmap_radius

    def init_general_heatmap(
        self,
        height: int,
        width: int,
        heatmap_radius: int,
        gaussian: bool,
    ):
        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, height * 2, device=self.device),
            torch.arange(0, width * 2, device=self.device)
        )

        if gaussian:
            self.general_heatmap = torch.exp(
                -0.5 * ((y_grid - height) ** 2 + (x_grid - width) ** 2)
                / (heatmap_radius ** 2)
            )
        else:
            self.general_heatmap = torch.zeros(
                (height * 2, width * 2),
                device=self.device
            )

        distance = (
            (y_grid - height) ** 2 +
            (x_grid - width) ** 2
        ).sqrt()

        mask = distance <= heatmap_radius

        self.general_heatmap[~mask] = 0


    def cut_out_rectanges(
        self,
        source: torch.Tensor,
        points: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, num_points, _ = points.size()
        source_height, source_width = source.size()

        x = points[:, :, 0].clamp(0, width - 1)
        y = points[:, :, 1].clamp(0, height - 1)

        x_start = (width - x).view(1, -1, 1, 1) \
            .expand(batch_size, num_points, source_height, width)

        y_start = (height - y).view(1, -1, 1, 1) \
            .expand(batch_size, num_points, height, width)

        x_indices = x_start + torch.arange(
            width, device=self.device
        ).view(1, 1, 1, -1)

        y_indices = y_start + torch.arange(
            height, device=self.device
        ).view(1, 1, -1, 1)

        source = source.view(1, 1, source_height, source_width) \
            .expand(batch_size, num_points, source_height, source_width)

        vertical_strips = torch.gather(source, 3, x_indices)

        rectangles = torch.gather(vertical_strips, 2, y_indices)

        return rectangles

    def forward(
        self,
        feature_maps: torch.Tensor,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_maps, height, width = feature_maps.size()
        num_points = num_maps // 3

        landmarks = landmarks.long()

        heatmaps = self.cut_out_rectanges(
            self.general_heatmap,
            landmarks,
            height,
            width,
        )

        offsetmap_x = self.cut_out_rectanges(
            self.general_offsetmap_x,
            landmarks,
            height,
            width,
        )

        offsetmap_y = self.cut_out_rectanges(
            self.general_offsetmap_y,
            landmarks,
            height,
            width,
        )

        heatmap_loss = F.binary_cross_entropy_with_logits(
            feature_maps[:, :num_points],
            heatmaps,
            reduction="none",
        ).mean(dim=(2, 3))

        offsetmap_x_loss = F.l1_loss(
            feature_maps[:, num_points:num_points * 2],
            offsetmap_x,
            reduction="none",
        ).mean(dim=(2, 3))

        offsetmap_y_loss = F.l1_loss(
            feature_maps[:, num_points * 2:],
            offsetmap_y,
            reduction="none",
        ).mean(dim=(2, 3))

        loss = 2 * heatmap_loss + offsetmap_x_loss + offsetmap_y_loss

        mask = (landmarks > 0).prod(-1)
        
        loss = (loss * mask).sum() / mask.sum()

        return loss
