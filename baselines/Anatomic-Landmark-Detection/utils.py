from __future__ import print_function, division
import torch
import numpy as np
import math


def get_statistical_results(offset, config):
    SDR = torch.zeros(config.landmarkNum, 5)
    SD = torch.zeros(config.landmarkNum)
    MRE = torch.mean(offset, 0)

    for landmarkId in range(config.landmarkNum):
        landmarkCol = offset[:, landmarkId].clone()
        train_mm = torch.tensor(
            [
                landmarkCol[landmarkCol <= 1].size()[0],
                landmarkCol[landmarkCol <= 2].size()[0],
                landmarkCol[landmarkCol <= 2.5].size()[0],
                landmarkCol[landmarkCol <= 3.0].size()[0],
                landmarkCol[landmarkCol <= 4.0].size()[0],
            ]
        ).float()
        SDR[landmarkId, :] = train_mm / landmarkCol.shape[0]
        SD[landmarkId] = torch.sqrt(
            torch.sum(torch.pow(landmarkCol - MRE[landmarkId], 2))
            / (landmarkCol.shape[0] - 1)
        )

    return SDR, SD, MRE


def regression_voting(heatmaps, R):
    # print("11", time.asctime())
    topN = int(R * R * 3.1415926)
    heatmap = heatmaps[0]
    imageNum, featureNum, h, w = heatmap.size()
    landmarkNum = int(featureNum / 3)
    heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)

    predicted_landmarks = torch.zeros((imageNum, landmarkNum, 2))
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
            VotePosi = (
                (topnXoff + topnYoff + indexs[imageId][landmarkId])
                .cpu()
                .numpy()
                .astype("int")
            )
            tem = VotePosi[VotePosi >= 0]
            maxid = 0
            if len(tem) > 0:
                maxid = np.argmax(np.bincount(tem))
            x = maxid // w
            y = maxid - x * w
            x, y = x / (h - 1), y / (w - 1)
            predicted_landmarks[imageId][landmarkId] = torch.Tensor([x, y])
    return predicted_landmarks


def Mydist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_deviation(coordinates1, lables, new_dataset):
    predictions = coordinates1.clone()
    targets = lables.clone()

    if new_dataset:
        predictions[:, :, 0] = predictions[:, :, 0] * 175.3481
        predictions[:, :, 1] = predictions[:, :, 1] * 237.23568

        targets[:, :, 0] = targets[:, :, 0] * 175.3481
        targets[:, :, 1] = targets[:, :, 1] * 237.23568

    else:
        predictions[:, :, 0] = predictions[:, :, 0] * 193.4
        predictions[:, :, 1] = predictions[:, :, 1] * 239.9

        targets[:, :, 0] = targets[:, :, 0] * 193.4
        targets[:, :, 1] = targets[:, :, 1] * 239.9

    print(predictions)
    print(targets)


    squared_difference = (predictions - targets) ** 2
    sum_squared_difference = squared_difference.sum(2)
    distance = sum_squared_difference.sqrt()

    print("Mean distance: ", distance.mean())

    return distance
