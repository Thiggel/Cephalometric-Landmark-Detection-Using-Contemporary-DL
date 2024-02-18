from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

from models.CephalometricLandmarkDetector import CephalometricLandmarkDetector
from models.baselines.yao import YaoLandmarkDetection


@dataclass
class ModelType:
    name: str
    resize_to: tuple[int, int]
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    ViT = "ViT"
    ConvNextV2 = "ConvNextV2"
    Yao = "Yao"

    @staticmethod
    def model_types():
        return {
            ModelTypes.ViT: ModelType(
                name='ViT',
                resize_to=(450, 450),
                model=CephalometricLandmarkDetector
            ),
            ModelTypes.ConvNextV2: ModelType(
                name='ConvNextV2',
                resize_to=(450, 450),
                model=CephalometricLandmarkDetector
            ),
            ModelTypes.Yao: ModelType(
                name='Yao',
                resize_to=(576, 512),
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
