from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np

from HeatmapHelper import HeatmapHelper
from OffsetmapHelper import OffsetmapHelper


class HeatmapOffsetmapLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # .use_gpu, R1, R2, image_scale, batchSize, landmarkNum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heatmap_helper = HeatmapHelper(
            resized_image_size=config.image_scale,
        )
        self.offsetmap_helper = OffsetmapHelper(
            resized_image_size=config.image_scale, offset_map_radius=41
        )
        self.R1 = config.R1
        self.width = config.image_scale[1]
        self.height = config.image_scale[0]
        self.imageNum = config.batchSize
        self.landmarkNum = config.landmarkNum

        self.binaryLoss = nn.BCEWithLogitsLoss(None, True)
        self.l1Loss = torch.nn.L1Loss()

        self.offsetMapx = torch.ones(
            (self.height * 2, self.width * 2), dtype=torch.float32, device=self.device
        )

        self.offsetMapy = torch.ones(
            (self.height * 2, self.width * 2), dtype=torch.float32, device=self.device
        )

        self.HeatMap = torch.zeros(
            (self.height * 2, self.width * 2), dtype=torch.float32, device=self.device
        )
        self.mask = torch.zeros(
            (self.height * 2, self.width * 2), dtype=torch.float32, device=self.device
        )

        self.offsetMapX_groundTruth = Variable(
            torch.zeros(
                self.imageNum,
                self.landmarkNum,
                self.height,
                self.width,
                device=self.device,
            )
        )

        self.offsetMapY_groundTruth = Variable(
            torch.zeros(
                self.imageNum,
                self.landmarkNum,
                self.height,
                self.width,
                device=self.device,
            )
        )

        self.binary_class_groundTruth1 = Variable(
            torch.zeros(
                self.imageNum,
                self.landmarkNum,
                self.height,
                self.width,
                device=self.device,
            )
        )

        self.binary_class_groundTruth2 = Variable(
            torch.zeros(
                self.imageNum,
                self.landmarkNum,
                self.height,
                self.width,
                device=self.device,
            )
        )

        self.offsetMask = Variable(
            torch.zeros(
                self.imageNum,
                self.landmarkNum,
                self.height,
                self.width,
                device=self.device,
            )
        )

        rr = config.R1
        referPoint = (self.height, self.width)
        for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                temdis = utils.Mydist(referPoint, (i, j))
                if temdis <= rr:
                    self.HeatMap[i][j] = 1
        rr = config.R2
        referPoint = (self.height, self.width)
        for i in range(referPoint[0] - rr, referPoint[0] + rr + 1):
            for j in range(referPoint[1] - rr, referPoint[1] + rr + 1):
                temdis = utils.Mydist(referPoint, (i, j))
                if temdis <= rr:
                    self.mask[i][j] = 1

        for i in range(2 * self.height):
            self.offsetMapx[i, :] = self.offsetMapx[i, :] * i

        for i in range(2 * self.width):
            self.offsetMapy[:, i] = self.offsetMapy[:, i] * i

        self.offsetMapx = referPoint[0] - self.offsetMapx
        self.offsetMapy = referPoint[1] - self.offsetMapy

        self.HeatMap = Variable(self.HeatMap)

        self.mask = Variable(self.mask)

        self.offsetMapx = Variable(self.offsetMapx) / config.R2

        self.offsetMapy = Variable(self.offsetMapy) / config.R2

        self.zeroTensor = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width),
            device=self.device,
        )

    def getOffsetMask(self, h, w, X, Y):
        for imageId in range(self.imageNum):
            for landmarkId in range(self.landmarkNum):
                self.offsetMask[imageId, landmarkId, :, :] = self.mask[
                    h - X[imageId][landmarkId] : 2 * h - X[imageId][landmarkId],
                    w - Y[imageId][landmarkId] : 2 * w - Y[imageId][landmarkId],
                ]
        return self.offsetMask

    def forward(self, featureMaps, landmarks):
        # TODO: paste back old logic and make everything 100% equal to the old implementation

        landmarks = landmarks.to(self.device)
        heatmaps, _ = self.heatmap_helper.create_heatmaps(
            landmarks * torch.tensor([self.width, self.height], device=self.device).unsqueeze(0).unsqueeze(0), 40
        )

        offsetmaps = self.offsetmap_helper.create_offset_maps(
            landmarks * torch.tensor([self.width, self.height], device=self.device).unsqueeze(0).unsqueeze(0)
        )
        binary_losses = self.binaryLoss(
            featureMaps[:, :self.landmarkNum, :, :],
            heatmaps
        )

        offset_x_losses = self.l1Loss(
            featureMaps[:, self.landmarkNum:self.landmarkNum * 2, :, :],
            offsetmaps[:, :, 0],
        )

        offset_y_losses = self.l1Loss(
            featureMaps[:, self.landmarkNum * 2:, :, :],
            offsetmaps[:, :, 1],
        )

        loss = (2 * binary_losses + offset_x_losses + offset_y_losses).mean()

        return loss
