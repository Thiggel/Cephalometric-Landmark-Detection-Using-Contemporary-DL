from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

from models.CephalometricLandmarkDetector import CephalometricLandmarkDetector
from models.baselines.yao import YaoLandmarkDetection


@dataclass
class ModelType:
    name: str
    crop: bool
    resize_to: tuple[int, int]
    use_heatmaps: bool
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    ViT = "ViT"
    ConvNextV2 = "ConvNextV2"
    Yao = "Yao"
    Kim = "Kim"

    @staticmethod
    def model_types():
        return {
            ModelTypes.Kim: ModelType(
                name='Kim',
                crop=True,
                resize_to=(1360, 1360),
                use_heatmaps=False,
                model=CephalometricLandmarkDetector
            ),
            ModelTypes.ViT: ModelType(
                name='ViT',
                crop=False,
                resize_to=(224, 224),
                use_heatmaps=False,
                model=CephalometricLandmarkDetector
            ),
            ModelTypes.ConvNextV2: ModelType(
                name='ConvNextV2',
                crop=False,
                resize_to=(450, 450),
                use_heatmaps=False,
                model=CephalometricLandmarkDetector
            ),
            ModelTypes.Yao: ModelType(
                name='Yao',
                crop=False,
                resize_to=(576, 512),
                use_heatmaps=True,
                model=YaoLandmarkDetection
            ),
        }

    @staticmethod
    def get_model_type(name: str) -> ModelType:
        return ModelTypes.model_types()[ModelTypes[name]]

    @staticmethod
    def get_model_types() -> list[str]:
        return list(map(
            lambda x: x.name,
            list(ModelTypes.model_types().keys())
        ))
