from dataclasses import dataclass
from typing import Callable
from enum import Enum
import torch.nn as nn

from models.ConvNextV2 import ConvNextV2
from models.ViT import ViT
from models.baselines.yao import LandmarkDetection


@dataclass
class ModelType:
    name: str
    resize_to: tuple[int, int]
    initialize: Callable[[str], nn.Module]


class ModelTypes(Enum):
    ViT = "ViT"
    ConvNextV2 = "ConvNextV2"
    Yao = "Yao"

    @staticmethod
    def model_types():
        return {
            ModelTypes.ViT: ModelType(
                'ViT',
                (450, 450),
                lambda model_type: ViT(model_type=model_type)
            ),
            ModelTypes.ConvNextV2: ModelType(
                'ConvNextV2',
                (450, 450),
                lambda model_type: ConvNextV2(model_type=model_type)
            ),
            ModelTypes.Yao: ModelType(
                'Yao',
                (576, 512),
                lambda _: LandmarkDetection()
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
