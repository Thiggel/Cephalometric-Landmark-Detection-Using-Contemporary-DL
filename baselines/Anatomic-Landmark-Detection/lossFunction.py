from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils

from HeatmapHelper import HeatmapHelper
from OffsetmapHelper import OffsetmapHelper


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

        self.width = config.image_scale[1]
        self.height = config.image_scale[0]
        self.imageNum = config.batchSize
        self.landmarkNum = config.landmarkNum

        self.binaryLoss = nn.BCEWithLogitsLoss(None, True)
        self.l1Loss = nn.L1Loss()

        self.general_offsetmap = self.get_general_offsetmap()
        self.general_heatmap = self.get_general_heatmap()

        self.offsetmap = torch.zeros(
            (2, self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device
        )

        self.heatmap = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device
        )

    def get_general_offsetmap(self):
        general_offsetmap = torch.ones(
            (2, self.height * 2, self.width * 2),
            device=self.device
        )

        general_offsetmap[0] *= torch.arange(2 * self.width)
        general_offsetmap[1] *= torch.arange(2 * self.height).unsqueeze(-1)

        general_offsetmap = torch.flip(general_offsetmap, dims=[1, 2])

        general_offsetmap /= self.offsetmap_radius

        return general_offsetmap

    def get_general_heatmap(self):
        general_heatmap = torch.zeros(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        radius = self.heatmap_radius

        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, self.height * 2, device=self.device),
            torch.arange(0, self.width * 2, device=self.device)
        )

        distance = (
            (y_grid - self.height) ** 2 +
            (x_grid - self.width) ** 2
        ).sqrt()

        mask = distance <= radius

        general_heatmap[mask] = 1

        return general_heatmap

    def forward(self, featureMaps, landmarks):
        h, w = featureMaps.size()[2:]
        landmarks = (
            landmarks.to(self.device) * torch.tensor([h, w], device=self.device)
        ).long()

        for imageId in range(self.imageNum):
            for landmarkId in range(self.landmarkNum):
                x = landmarks[imageId, landmarkId, 0]
                y = landmarks[imageId, landmarkId, 1]

                self.heatmap[imageId, landmarkId, :, :] = self.general_heatmap[
                    h - x : 2 * h - x,
                    w - y: 2 * w - y,
                ]

                self.offsetmap[1, imageId, landmarkId, :, :] = self.general_offsetmap[
                    1,
                    h - x : 2 * h - x,
                    w - y : 2 * w - y,
                ]

                self.offsetmap[0, imageId, landmarkId, :, :] = self.general_offsetmap[
                    0,
                    h - x : 2 * h - x,
                    w - y : 2 * w - y,
                ]

        indexs = self.heatmap > 0
        temloss = [
            (
                [
                    2
                    * self.binaryLoss(
                        featureMaps[imageId][landmarkId],
                        self.heatmap[imageId][landmarkId],
                    ),
                    self.l1Loss(
                        featureMaps[imageId][landmarkId + self.landmarkNum * 1][
                            indexs[imageId][landmarkId]
                        ],
                        self.offsetmap[1, imageId, landmarkId][
                            indexs[imageId][landmarkId]
                        ],
                    ),
                    self.l1Loss(
                        featureMaps[imageId][landmarkId + self.landmarkNum * 2][
                            indexs[imageId][landmarkId]
                        ],
                        self.offsetmap[0, imageId, landmarkId][
                            indexs[imageId][landmarkId]
                        ],
                    ),
                ]
            )
            for imageId in range(self.imageNum)
            for landmarkId in range(self.landmarkNum)
        ]

        loss = (
            sum([sum(temloss[ind]) for ind in range(self.imageNum * self.landmarkNum)])
        ) / (self.imageNum * self.landmarkNum)

        return loss
