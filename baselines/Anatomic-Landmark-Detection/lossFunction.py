from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class HeatmapOffsetmapLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.radius1 = config.R1
        self.radius2 = config.R2
        self.width = config.image_scale[1]
        self.height = config.image_scale[0]
        self.num_images = config.batchSize
        self.num_landmarks = config.landmarkNum

        self.binary_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = torch.nn.L1Loss()

        self.initialize_ground_truth()
        self.initialize_heatmap_and_masks()

    def initialize_ground_truth(self):
        size = (self.num_images, self.num_landmarks, self.height, self.width)
        self.offset_map_x_gt = Variable(torch.zeros(size, device=self.device))
        self.offset_map_y_gt = Variable(torch.zeros(size, device=self.device))
        self.binary_class_gt = Variable(torch.zeros(size, device=self.device))

    def initialize_heatmap_and_masks(self):
        self.heatmap = torch.zeros(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        self.mask = torch.zeros(
            (self.height * 2, self.width * 2),
            device=self.device
        )

        yy, xx = torch.meshgrid(
            torch.arange(
                0,
                self.height * 2,
                device=self.device
            ),
            torch.arange(
                0,
                self.width * 2,
                device=self.device
            )
        )

        refer_point = torch.tensor(
            [self.height, self.width],
            dtype=torch.float32,
            device=self.device
        )

        heatmap_distance = torch.sqrt(
            (yy - refer_point[0]) ** 2 + (xx - refer_point[1]) ** 2
        )

        self.heatmap[heatmap_distance <= self.radius1] = 1

        mask_distance = torch.sqrt(
            (yy - refer_point[0]) ** 2 + (xx - refer_point[1]) ** 2
        )
        self.mask[mask_distance <= self.radius2] = 1

        self.offset_map_x = (refer_point[0] - xx) / self.radius2
        self.offset_map_y = (refer_point[1] - yy) / self.radius2

    def forward(self, feature_maps, landmarks):
        h, w = feature_maps.size()[2], feature_maps.size()[3]
        X = ((landmarks[:, :, 0] * (h - 1)).long()).int()
        Y = ((landmarks[:, :, 1] * (w - 1)).long()).int()

        self.binary_class_gt.scatter_(2, X.view(self.num_images, self.num_landmarks, 1, 1), self.heatmap[h - X: 2 * h - X, w - Y: 2 * w - Y])
        self.offset_map_x_gt.scatter_(2, X.view(self.num_images, self.num_landmarks, 1, 1), self.offset_map_x[h - X: 2 * h - X, w - Y: 2 * w - Y])
        self.offset_map_y_gt.scatter_(2, X.view(self.num_images, self.num_landmarks, 1, 1), self.offset_map_y[h - X: 2 * h - X, w - Y: 2 * w - Y])

        index_mask = self.binary_class_gt > 0

        binary_losses = 2 * self.binary_loss(
            feature_maps,
            self.binary_class_gt
        )

        offset_x_losses = self.l1_loss(
            feature_maps[:, self.num_landmarks:self.num_landmarks*2][index_mask],
            self.offset_map_x_gt[index_mask]
        )

        offset_y_losses = self.l1_loss(
            feature_maps[:, self.num_landmarks*2:][index_mask],
            self.offset_map_y_gt[index_mask]
        )

        loss = (binary_losses + offset_x_losses + offset_y_losses).mean()

        return loss
