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
        #self.use_gpu = config.use_gpu
        self.R1 = config.R1
        self.width = config.image_scale[1]
        self.height = config.image_scale[0]
        self.imageNum = config.batchSize
        self.landmarkNum = config.landmarkNum

        self.binaryLoss = nn.BCEWithLogitsLoss(None, True).to(self.device)
        self.l1Loss = torch.nn.L1Loss().to(self.device)

        self.offsetMapx = np.ones((self.height * 2, self.width * 2))
        self.offsetMapy = np.ones((self.height * 2, self.width * 2))

        self.HeatMap = np.zeros((self.height * 2, self.width * 2))
        self.mask = np.zeros((self.height * 2, self.width * 2))

        self.offsetMapX_groundTruth = Variable(
            torch.zeros(self.imageNum, self.landmarkNum, self.height, self.width).to(
                self.device
            )
        )
        self.offsetMapY_groundTruth = Variable(
            torch.zeros(self.imageNum, self.landmarkNum, self.height, self.width).to(
                self.device
            )
        )
        self.binary_class_groundTruth1 = Variable(
            torch.zeros(self.imageNum, self.landmarkNum, self.height, self.width).to(
                self.device
            )
        )
        self.binary_class_groundTruth2 = Variable(
            torch.zeros(self.imageNum, self.landmarkNum, self.height, self.width).to(
                self.device
            )
        )
        self.offsetMask = Variable(
            torch.zeros(self.imageNum, self.landmarkNum, self.height, self.width).to(
                self.device
            )
        )

        rr = config.R1
        dev = 4
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
        self.HeatMap = (
            Variable(torch.from_numpy(self.HeatMap)).to(self.device).float()
        )
        self.mask = Variable(torch.from_numpy(self.mask)).to(self.device).float()
        self.offsetMapx = (
            Variable(torch.from_numpy(self.offsetMapx)).to(self.device).float()
            / config.R2
        )
        self.offsetMapy = (
            Variable(torch.from_numpy(self.offsetMapy)).to(self.device).float()
            / config.R2
        )

        self.zeroTensor = torch.zeros(
            (self.imageNum, self.landmarkNum, self.height, self.width)
        ).to(self.device)

        return

    def getOffsetMask(self, h, w, X, Y):
        for imageId in range(self.imageNum):
            for landmarkId in range(self.landmarkNum):
                self.offsetMask[imageId, landmarkId, :, :] = self.mask[
                    h - X[imageId][landmarkId] : 2 * h - X[imageId][landmarkId],
                    w - Y[imageId][landmarkId] : 2 * w - Y[imageId][landmarkId],
                ]
        return self.offsetMask

    def forward(self, featureMaps, landmarks):
        landmarks = landmarks.to(self.device)
        heatmaps, _ = self.heatmap_helper.create_heatmaps(
            landmarks * torch.tensor([self.height, self.width], device=self.device).unsqueeze(0).unsqueeze(0), 40
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

        
        h, w = featureMaps.size()[2], featureMaps.size()[3]
        X = np.clip(
            np.round((landmarks[:, :, 0] * (h - 1)).numpy()).astype("int"), 0, h - 1
        )
        Y = np.clip(
            np.round((landmarks[:, :, 1] * (w - 1)).numpy()).astype("int"), 0, w - 1
        )
        binary_class_groundTruth = self.binary_class_groundTruth1

        for imageId in range(self.imageNum):
            for landmarkId in range(self.landmarkNum):
                # ~ self.binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[h - X[imageId][landmarkId]: 2*h - X[imageId][landmarkId], w - Y[imageId][landmarkId]: 2*w - Y[imageId][landmarkId]]
                binary_class_groundTruth[imageId, landmarkId, :, :] = self.HeatMap[
                    h - X[imageId][landmarkId] : 2 * h - X[imageId][landmarkId],
                    w - Y[imageId][landmarkId] : 2 * w - Y[imageId][landmarkId],
                ]
                self.offsetMapX_groundTruth[imageId, landmarkId, :, :] = (
                    self.offsetMapx[
                        h - X[imageId][landmarkId] : 2 * h - X[imageId][landmarkId],
                        w - Y[imageId][landmarkId] : 2 * w - Y[imageId][landmarkId],
                    ]
                )
                self.offsetMapY_groundTruth[imageId, landmarkId, :, :] = (
                    self.offsetMapy[
                        h - X[imageId][landmarkId] : 2 * h - X[imageId][landmarkId],
                        w - Y[imageId][landmarkId] : 2 * w - Y[imageId][landmarkId],
                    ]
                )


        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 2)

        ax[0, 0].imshow(binary_class_groundTruth[0, 0, :, :].cpu().detach().numpy())
        ax[0, 1].imshow(heatmaps[0, 0, :, :].cpu().detach().numpy())

        ax[1, 0].imshow(self.offsetMapX_groundTruth[0, 0, :, :].cpu().detach().numpy())
        ax[1, 1].imshow(offsetmaps[0, 0, 1, :, :].cpu().detach().numpy())

        ax[2, 0].imshow(self.offsetMapY_groundTruth[0, 0, :, :].cpu().detach().numpy())
        ax[2, 1].imshow(offsetmaps[0, 0, 0, :, :].cpu().detach().numpy())

        ax[0, 0].scatter(Y[0, 0], X[0, 0])
        ax[0, 1].scatter(Y[0, 0], X[0, 0])
        ax[1, 0].scatter(Y[0, 0], X[0, 0])
        ax[1, 1].scatter(Y[0, 0], X[0, 0])
        ax[2, 0].scatter(Y[0, 0], X[0, 0])
        ax[2, 1].scatter(Y[0, 0], X[0, 0])

        plt.show()

        indexs = binary_class_groundTruth > 0
        temloss = [
            (
                [
                    2
                    * self.binaryLoss(
                        featureMaps[imageId][landmarkId],
                        binary_class_groundTruth[imageId][landmarkId],
                    ),
                    self.l1Loss(
                        featureMaps[imageId][landmarkId + self.landmarkNum * 1][
                            indexs[imageId][landmarkId]
                        ],
                        self.offsetMapX_groundTruth[imageId][landmarkId][
                            indexs[imageId][landmarkId]
                        ],
                    ),
                    self.l1Loss(
                        featureMaps[imageId][landmarkId + self.landmarkNum * 2][
                            indexs[imageId][landmarkId]
                        ],
                        self.offsetMapY_groundTruth[imageId][landmarkId][
                            indexs[imageId][landmarkId]
                        ],
                    ),
                ]
                if X[imageId][landmarkId] > 0 and Y[imageId][landmarkId] > 0
                else [0, 0, 0]
            )
            for imageId in range(self.imageNum)
            for landmarkId in range(self.landmarkNum)
        ]

        loss = (
            sum([sum(temloss[ind]) for ind in range(self.imageNum * self.landmarkNum)])
        ) / (self.imageNum * self.landmarkNum)

        return loss
