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

        self.general_offsetmap_x = self.get_general_offsetmap_x()

        self.general_offsetmap_y = self.get_general_offsetmap_y()

        self.general_heatmap = self.get_general_heatmap()

        self.offsetmap_x = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device
        )

        self.offsetmap_y = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device
        )

        self.heatmap = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device
        )

    def get_general_offsetmap_x(self):
        general_offsetmap_x = torch.ones(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        for i in range(2 * self.height):
            general_offsetmap_x[i, :] *= i

        general_offsetmap_x = self.height - general_offsetmap_x

        general_offsetmap_x /= self.offsetmap_radius

        return general_offsetmap_x

    def get_general_offsetmap_y(self):
        general_offsetmap_y = torch.ones(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        for i in range(2 * self.width):
            general_offsetmap_y[:, i] *= i

        general_offsetmap_y = self.width - general_offsetmap_y

        general_offsetmap_y /= self.offsetmap_radius

        return general_offsetmap_y

    def get_general_heatmap(self):
        general_heatmap = torch.zeros(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        radius = self.heatmap_radius

        x_indices = torch.arange(
            self.height - radius,
            self.height + radius,
            device=self.device
        )

        y_indices = torch.arange(
            self.width - radius,
            self.width + radius,
            device=self.device
        )

        distance = (
            (x_indices - self.height) ** 2 +
            (y_indices - self.width) ** 2
        ).sqrt()

        mask = distance <= radius

        #general_heatmap[
        #    x_indices[mask],
        #    y_indices[mask]
        #] = 1

        rr = radius

        for i in range(self.height - rr, self.height + rr + 2):
            for j in range(self.width - rr, self.width + rr + 2):
                temdis = utils.Mydist((self.height, self.width), (i, j))
                if temdis <= rr:
                    general_heatmap[i][j] = 1

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

                self.offsetmap_x[imageId, landmarkId, :, :] = self.general_offsetmap_x[
                    h - x : 2 * h - x,
                    w - y : 2 * w - y,
                ]

                self.offsetmap_y[imageId, landmarkId, :, :] = self.general_offsetmap_y[
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
                        self.offsetmap_x[imageId][landmarkId][
                            indexs[imageId][landmarkId]
                        ],
                    ),
                    self.l1Loss(
                        featureMaps[imageId][landmarkId + self.landmarkNum * 2][
                            indexs[imageId][landmarkId]
                        ],
                        self.offsetmap_y[imageId][landmarkId][
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
