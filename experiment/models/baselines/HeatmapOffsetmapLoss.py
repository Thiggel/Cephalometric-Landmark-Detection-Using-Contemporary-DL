from __future__ import print_function, division
import torch
import torch.nn as nn


class HeatmapOffsetmapLoss(nn.Module):
    def __init__(
        self,
        config,
        heatmap_radius: int = 40,
        offsetmap_radius: int = 40,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.heatmap_radius = heatmap_radius
        self.offsetmap_radius = offsetmap_radius

        self.binary_loss = nn.BCEWithLogitsLoss(None, True)
        self.l1_loss = nn.L1Loss()

    def init_offset_and_heatmaps(
        self,
        batch_size: int,
        num_points: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self, "offsetmap_x"):
            self.offsetmap_x = torch.zeros(
                (batch_size, num_points, height, width),
                device=self.device
            )

        if not hasattr(self, "offsetmap_y"):
            self.offsetmap_y = torch.zeros(
                (batch_size, num_points, height, width),
                device=self.device
            ) 

        if not hasattr(self, "heatmap"):
            self.heatmap = torch.zeros(
                (batch_size, num_points, height, width),
                device=self.device
            )

    def init_general_offsetmap_x(self, height: int, width: int):
        if hasattr(self, "general_offsetmap_x"):
            pass 
         
        self.general_offsetmap_x = torch.ones(
            (height * 2, width * 2),
            device=self.device
        )

        for i in range(2 * height):
            self.general_offsetmap_x[i, :] *= i

        self.general_offsetmap_x = height - self.general_offsetmap_x

        self.general_offsetmap_x /= self.offsetmap_radius

    def init_general_offsetmap_y(self, height: int, width: int):
        if hasattr(self, "general_offsetmap_y"):
            pass

        self.general_offsetmap_y = torch.ones(
            (height * 2, width * 2),
            device=self.device
        )

        for i in range(2 * width):
            self.general_offsetmap_y[:, i] *= i

        self.general_offsetmap_y = width - self.general_offsetmap_y

        self.general_offsetmap_y /= self.offsetmap_radius

    def init_general_heatmap(self, height: int, width: int):
        if hasattr(self, "general_heatmap"):
            pass

        self.general_heatmap = torch.zeros(
            (height * 2, width * 2),
            device=self.device
        )

        radius = self.heatmap_radius

        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, height * 2, device=self.device),
            torch.arange(0, width * 2, device=self.device)
        )

        distance = (
            (y_grid - height) ** 2 +
            (x_grid - width) ** 2
        ).sqrt()

        mask = distance <= radius

        self.general_heatmap[mask] = 1

    def forward(
        self,
        feature_maps: torch.Tensor, 
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_points, h, w = feature_maps.size()
        num_points = num_points // 3
        landmarks = (
            landmarks.to(self.device) * torch.tensor([h, w], device=self.device)
        ).long()

        self.init_general_heatmap(h, w)
        self.init_general_offsetmap_x(h, w)
        self.init_general_offsetmap_y(h, w)
        self.init_offset_and_heatmaps(batch_size, num_points, h, w)

        for image_id in range(batch_size):
            for landmark_id in range(num_points):
                x = landmarks[image_id, landmark_id, 0]
                y = landmarks[image_id, landmark_id, 1]

                self.heatmap[image_id, landmark_id, :, :] = self.general_heatmap[
                    h - x : 2 * h - x,
                    w - y: 2 * w - y,
                ]

                self.offsetmap_x[image_id, landmark_id, :, :] = self.general_offsetmap_x[
                    h - x : 2 * h - x,
                    w - y : 2 * w - y,
                ]

                self.offsetmap_y[image_id, landmark_id, :, :] = self.general_offsetmap_y[
                    h - x : 2 * h - x,
                    w - y : 2 * w - y,
                ]

        indexs = self.heatmap > 0
        losses = [
            (
                [
                    2 * self.binary_loss(
                        feature_maps[image_id][landmark_id],
                        self.heatmap[image_id][landmark_id],
                    ),
                    self.l1_loss(
                        feature_maps[image_id][landmark_id + num_points * 1][
                            indexs[image_id][landmark_id]
                        ],
                        self.offsetmap_x[image_id, landmark_id][
                            indexs[image_id][landmark_id]
                        ],
                    ),
                    self.l1_loss(
                        feature_maps[image_id][landmark_id + num_points * 2][
                            indexs[image_id][landmark_id]
                        ],
                        self.offsetmap_y[image_id, landmark_id][
                            indexs[image_id][landmark_id]
                        ],
                    ),
                ]
            )
            for image_id in range(batch_size)
            for landmark_id in range(num_points)
        ]

        loss = sum([
            sum(losses[i]) for i in range(batch_size * num_points)
        ]) / (batch_size * num_points)

        return loss
