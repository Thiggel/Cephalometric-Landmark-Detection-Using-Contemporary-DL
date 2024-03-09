import torch
from lossFunction import HeatmapOffsetmapLoss
import argparse

loss = HeatmapOffsetmapLoss(
    argparse.Namespace(
        **{
            "R1": 41,
            "R2": 41,
            "image_scale": (800, 640),
            "batchSize": 2,
            "landmarkNum": 19,
        }
    )
)

test_feature_maps = torch.rand(2, 3, 800, 640)

min_value = torch.tensor([0.1, 0.1])
max_value = torch.tensor([0.9, 0.9])
landmarks = min_value + (max_value - min_value) * torch.rand(2, 19, 2)

print(landmarks)

loss(test_feature_maps, landmarks)
