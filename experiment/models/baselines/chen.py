from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from models.baselines.HeatmapOffsetmapLoss import HeatmapOffsetmapLoss
from models.losses.MaskedWingLoss import MaskedWingLoss


class dilationInceptionModule(nn.Module):
    def __init__(self, inplanes, planes):
        super(dilationInceptionModule, self).__init__()

        fnum = int(planes / 4)
        self.temConv1 = nn.Sequential(
            nn.Conv2d(inplanes, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.temConv2 = nn.Sequential(
            nn.Conv2d(inplanes, fnum, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.temConv3 = nn.Sequential(
            nn.Conv2d(
                inplanes, fnum, kernel_size=(3, 3), stride=1, padding=2, dilation=2
            ),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.temConv4 = nn.Sequential(
            nn.Conv2d(
                inplanes, fnum, kernel_size=(3, 3), stride=1, padding=4, dilation=4
            ),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        x1 = self.temConv1(x)
        x2 = self.temConv2(x)
        x3 = self.temConv3(x)
        x4 = self.temConv4(x)
        y = torch.cat((x1, x2, x3, x4), 1)
        return y


class fusionResNet50(nn.Module):
    def __init__(self, model, batchSize, landmarksNum, useGPU, image_scale, R):
        super(fusionResNet50, self).__init__()

        para_list = list(model.children())
        self.relu = nn.ReLU(inplace=True)
        self.resnet_layer1 = nn.Sequential(*para_list[:5])
        self.resnet_layer2 = para_list[5]
        self.resnet_layer3 = para_list[6]
        self.resnet_layer4 = para_list[7]

        fnum = 96
        self.fnum = fnum
        self.f_conv4 = nn.Sequential(
            nn.Conv2d(2048, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv3 = nn.Sequential(
            nn.Conv2d(1024, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv2 = nn.Sequential(
            nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv1 = nn.Sequential(
            nn.Conv2d(256, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.avgPool8t = nn.AvgPool2d(8, 8)
        self.avgPool4t = nn.AvgPool2d(4, 4)
        self.avgPool2t = nn.AvgPool2d(2, 2)
        self.attentionLayer1 = nn.Sequential(
            nn.Linear(500, 128, bias=False),
            nn.BatchNorm1d(1, track_running_stats=False),
            nn.Tanh(),
            nn.Linear(128, landmarksNum * 3, bias=False),
            # ~ nn.BatchNorm1d(1,track_running_stats=False),
            nn.Softmax(dim=0),
        )

        moduleList = []
        for i in range(landmarksNum * 3):
            # ~ temConv = dilationInceptionModule(fnum*4, 1)
            temConv = nn.Conv2d(fnum * 4, 1, kernel_size=(1, 1), stride=1, padding=0)
            moduleList.append(temConv)

        self.moduleList = nn.ModuleList(moduleList)

        scaleFactorList = []
        for i in range(landmarksNum * 3):
            scaleFactorList.append(nn.Linear(1, 1, bias=False))
        self.scaleFactorList = nn.ModuleList(scaleFactorList)

        self.inception = dilationInceptionModule(fnum * 4, fnum * 4)
        self.prediction = nn.Conv2d(
            2048, landmarksNum * 3, kernel_size=(1, 1), stride=1, padding=0
        )
        self.Upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.Upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.Upsample8 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.Upsample16 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.Upsample32 = nn.Upsample(scale_factor=32, mode="bilinear")

        self.landmarksNum = landmarksNum
        self.batchSize = batchSize
        self.useGPU = useGPU
        self.R2 = R

    def getCoordinate(self, outputs1):
        heatmaps = F.sigmoid(outputs1[:, 0 : self.landmarksNum, :, :])
        heatmap_sum = torch.sum(
            heatmaps.view(self.batchSize, self.landmarksNum, -1), dim=2
        )

        Xmap1 = heatmaps * self.coordinateX
        Ymap1 = heatmaps * self.coordinateY

        Xmean1 = (
            torch.sum(Xmap1.view(self.batchSize, self.landmarksNum, -1), dim=2)
            / heatmap_sum
        )
        Ymean1 = (
            torch.sum(Ymap1.view(self.batchSize, self.landmarksNum, -1), dim=2)
            / heatmap_sum
        )

        coordinateMean1 = torch.stack([Xmean1, Ymean1]).permute(1, 2, 0)
        coordinateMean2 = 0

        XDevmap = torch.pow(
            self.coordinateX - Xmean1.view(self.batchSize, self.landmarksNum, 1, 1), 2
        )
        YDevmap = torch.pow(
            self.coordinateY - Ymean1.view(self.batchSize, self.landmarksNum, 1, 1), 2
        )

        XDevmap = heatmaps * XDevmap
        YDevmap = heatmaps * YDevmap

        coordinateDev = (
            torch.sum(
                (XDevmap + YDevmap).view(self.batchSize, self.landmarksNum, -1), dim=2
            )
            / heatmap_sum
        )

        return coordinateMean1, coordinateMean2, coordinateDev

    def getAttention(self, bone, fnum):
        bone = self.avgPool8t(bone).view(fnum, -1)
        bone = bone.unsqueeze(1)
        y = self.attentionLayer1(bone).squeeze(1).transpose(1, 0)

        return y

    def predictionWithAttention(self, bone, attentions):
        featureNum, channelNum = attentions.size()[0], attentions.size()[1]
        attentionMaps = []
        for i in range(featureNum):
            attention = attentions[i, :]
            attention = attention.view(1, channelNum, 1, 1)
            attentionMap = attention * bone * channelNum
            attentionMaps.append(self.moduleList[i](attentionMap))
        attentionMaps = torch.stack(attentionMaps).squeeze().unsqueeze(0)
        return attentionMaps

    def forward(self, x):
        x = self.resnet_layer1(x)
        f1 = self.f_conv1(x)
        # ~ print(x.size())
        x = self.resnet_layer2(x)
        f2 = self.f_conv2(x)
        # ~ print(x.size())
        x = self.resnet_layer3(x)
        f3 = self.f_conv3(x)
        # ~ print(x.size())
        x = self.resnet_layer4(x)
        f4 = self.f_conv4(x)

        f2 = self.Upsample2(f2)
        f3 = self.Upsample4(f3)
        f4 = self.Upsample8(f4)

        bone = torch.cat((f1, f2, f3, f4), 1)
        bone = self.inception(bone)
        attention = self.getAttention(bone, self.fnum * 4)

        y = self.Upsample4(self.predictionWithAttention(bone, attention))
        coordinateMean1, coordinateMean2 = 0, 0

        return [y], coordinateMean1, coordinateMean2


class fusionVGG19(nn.Module):
    def __init__(
        self,
        model, 
        batch_size,
        num_points,
        resized_image_size,
    ):
        super(fusionVGG19, self).__init__()
        para_list = list(model.children())[0]

        self.VGG_layer1 = nn.Sequential(*para_list[:14])
        self.VGG_layer2 = nn.Sequential(*para_list[14:27])
        self.VGG_layer3 = nn.Sequential(*para_list[27:40])
        self.VGG_layer4 = nn.Sequential(*para_list[40:])
        self.relu = nn.ReLU(inplace=True)

        fnum = 64
        self.fnum = fnum
        self.f_conv4 = nn.Sequential(
            nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv3 = nn.Sequential(
            nn.Conv2d(512, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv2 = nn.Sequential(
            nn.Conv2d(256, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.f_conv1 = nn.Sequential(
            nn.Conv2d(128, fnum, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(fnum, track_running_stats=False),
            nn.ReLU(True),
        )

        self.avgPool8t = nn.AvgPool2d(8, 8)
        self.avgPool4t = nn.AvgPool2d(4, 4)
        self.avgPool2t = nn.AvgPool2d(2, 2)
        self.attentionLayer1 = nn.Sequential(
            nn.Linear(500, 128, bias=False),
            nn.BatchNorm1d(1, track_running_stats=False),
            nn.Tanh(),
            nn.Linear(128, num_points * 3, bias=False),
            nn.Softmax(dim=0),
        )

        moduleList = []
        for i in range(num_points * 3):
            temConv = nn.Conv2d(fnum * 4, 1, kernel_size=(1, 1), stride=1, padding=0)
            moduleList.append(temConv)

        self.moduleList = nn.ModuleList(moduleList)
        self.dilated_block = dilationInceptionModule(fnum * 4, fnum * 4)
        self.prediction = nn.Conv2d(
            fnum * 4, num_points * 3, kernel_size=(1, 1), stride=1, padding=0
        )
        self.Upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.Upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.Upsample8 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.Upsample16 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.Upsample32 = nn.Upsample(scale_factor=32, mode="bilinear")

        self.landmarksNum = num_points
        self.batchSize = batch_size
        self.R2 = 40

        self.height, self.width = resized_image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.coordinateX = torch.ones(
            self.batchSize, self.landmarksNum, self.height, self.width
        ).to(self.device)
        self.coordinateY = torch.ones(
            self.batchSize, self.landmarksNum, self.height, self.width
        ).to(self.device)

        for i in range(self.height):
            self.coordinateX[:, :, i, :] = self.coordinateX[:, :, i, :] * i

        for i in range(self.width):
            self.coordinateY[:, :, :, i] = self.coordinateY[:, :, :, i] * i

        self.coordinateX, self.coordinateY = self.coordinateX / (
            self.height - 1
        ), self.coordinateY / (self.width - 1)

    def getCoordinate(self, outputs1):
        heatmaps = F.sigmoid(outputs1[:, 0 : self.landmarksNum, :, :])
        heatmap_sum = torch.sum(
            heatmaps.view(self.batchSize, self.landmarksNum, -1), dim=2
        )

        Xmap1 = heatmaps * self.coordinateX
        Ymap1 = heatmaps * self.coordinateY

        Xmean1 = (
            torch.sum(Xmap1.view(self.batchSize, self.landmarksNum, -1), dim=2)
            / heatmap_sum
        )
        Ymean1 = (
            torch.sum(Ymap1.view(self.batchSize, self.landmarksNum, -1), dim=2)
            / heatmap_sum
        )

        coordinateMean1 = torch.stack([Xmean1, Ymean1]).permute(1, 2, 0)
        coordinateMean2 = 0

        XDevmap = torch.pow(
            self.coordinateX - Xmean1.view(self.batchSize, self.landmarksNum, 1, 1), 2
        )
        YDevmap = torch.pow(
            self.coordinateY - Ymean1.view(self.batchSize, self.landmarksNum, 1, 1), 2
        )

        XDevmap = heatmaps * XDevmap
        YDevmap = heatmaps * YDevmap

        coordinateDev = (
            torch.sum(
                (XDevmap + YDevmap).view(self.batchSize, self.landmarksNum, -1), dim=2
            )
            / heatmap_sum
        )

        return coordinateMean1, coordinateMean2, coordinateDev

    def getAttention(self, bone, fnum):
        bone = self.avgPool8t(bone).view(fnum, -1)
        bone = bone.unsqueeze(1)
        y = self.attentionLayer1(bone).squeeze(1).transpose(1, 0)
        return y

    def predictionWithAttention(self, bone, attentions):
        featureNum, channelNum = attentions.size()[0], attentions.size()[1]

        attentionMaps = []
        for i in range(featureNum):
            attention = attentions[i, :]
            attention = attention.view(1, channelNum, 1, 1)
            attentionMap = attention * bone * channelNum
            attentionMaps.append(self.moduleList[i](attentionMap))

        attentionMaps = torch.stack(attentionMaps).squeeze().unsqueeze(0)
        return attentionMaps

    def forward(self, x):
        x = self.VGG_layer1(x)
        f1 = self.f_conv1(x)

        x = self.VGG_layer2(x)
        f2 = self.f_conv2(x)

        x = self.VGG_layer3(x)
        f3 = self.f_conv3(x)

        x = self.VGG_layer4(x)
        f4 = self.f_conv4(x)

        f2 = self.Upsample2(f2)
        f3 = self.Upsample4(f3)
        f4 = self.Upsample8(f4)
        bone = torch.cat((f1, f2, f3, f4), 1)

        # Attentive Feature Pyramid Fusion
        bone = self.dilated_block(bone)
        attention = self.getAttention(bone, self.fnum * 4)
        y = self.Upsample4(self.predictionWithAttention(bone, attention))

        # predicting landmarks with the integral operation
        # coordinateMean1, coordinateMean2, coordinateDev = self.getCoordinate(y)

        return y


class ChenLandmarkPrediction(L.LightningModule):
    def __init__(
        self,
        point_ids: list[str],
        batch_size: int = 32,
        output_size: int = 19,
        original_image_size: int = (1920, 1080),
        resized_image_size: int = (800, 640),
        px_to_mm: int = 0.1,
        reduce_lr_patience: int = 25,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = fusionVGG19(
            torchvision.models.vgg19_bn(pretrained=True),
            batch_size,
            output_size,
            resized_image_size,
        )

        self.loss = HeatmapOffsetmapLoss()

        self.mm_error = MaskedWingLoss(
            px_to_mm=px_to_mm,
            original_image_size=original_image_size,
            resized_image_size=resized_image_size,
        )

        self.point_ids = point_ids

        self.reduce_lr_patience = reduce_lr_patience

    def forward_with_heatmaps(self, x):
        x = x.repeat(1, 3, 1, 1)
        output = self.model(x)
        return output

    def forward(self, x):
        output = self.forward_with_heatmaps(x)

        return self.get_points(output)

    def get_points(self, model_output: torch.Tensor):
        return self.regression_voting([model_output], 41)

    def regression_voting(self, heatmaps, R):
        # print("11", time.asctime())
        topN = int(R * R * 3.1415926)
        heatmap = heatmaps[0]
        imageNum, featureNum, h, w = heatmap.size()
        landmarkNum = int(featureNum / 3)
        heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)

        predicted_landmarks = torch.zeros(
            (imageNum, landmarkNum, 2),
            device=self.device
        )

        Pmap = heatmap[:, 0:landmarkNum, :].data
        Xmap = torch.round(heatmap[:, landmarkNum : landmarkNum * 2, :].data * R).long() * w
        Ymap = torch.round(heatmap[:, landmarkNum * 2 : landmarkNum * 3, :].data * R).long()
        topkP, indexs = torch.topk(Pmap, topN)
        # ~ plt.imshow(Pmap.reshape(imageNum, landmarkNum, h,w)[0][0], cmap='gray', interpolation='nearest')
        for imageId in range(imageNum):
            for landmarkId in range(landmarkNum):

                topnXoff = Xmap[imageId][landmarkId][
                    indexs[imageId][landmarkId]
                ]  # offset in x direction
                topnYoff = Ymap[imageId][landmarkId][
                    indexs[imageId][landmarkId]
                ]  # offset in y direction

                VotePosi = topnXoff + topnYoff + indexs[imageId][landmarkId]
                
                tem = VotePosi[VotePosi >= 0]
                maxid = 0
                if len(tem) > 0:
                    maxid = torch.argmax(torch.bincount(tem))
                x = maxid // w
                y = maxid - x * w
                x, y = x / (h - 1), y / (w - 1)
                predicted_landmarks[imageId][landmarkId] = torch.tensor([y, x], device=self.device)
        return predicted_landmarks

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        inputs, targets = batch

        predictions = self.forward_with_heatmaps(inputs)

        loss = self.loss(
            predictions,
            targets,
        )

        point_predictions = self.get_points(predictions)

        unreduced_mm_error = None

        if with_mm_error:

            unreduced_mm_error = self.mm_error.mm_error(
                point_predictions,
                targets,
                (targets > 0).prod(dim=-1)
            )

        return loss, unreduced_mm_error, point_predictions, targets

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _, _, _ = self.step(batch)

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error, _, _ = self.step(batch, with_mm_error=True)

        mm_error = mm_error.mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        (
            loss,
            mm_error,
            predictions,
            targets
        ) = self.step(batch, with_mm_error=True)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean())

        mm_error = mm_error.mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error, prog_bar=True)

        self.log(
            'percent_under_1mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 1)
        )
        self.log(
            'percent_under_2mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 2)
        )
        self.log(
            'percent_under_3mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 3)
        )
        self.log(
            'percent_under_4mm',
            self.mm_error.percent_under_n_mm(predictions, targets, 4)
        )

        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1.0
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    patience=self.reduce_lr_patience
                ),
                'monitor': 'val_loss'
            },
        }

