from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

from models.DirectPointPredictionBasedLandmarkDetection import \
    DirectPointPredictionBasedLandmarkDetection
from models.baselines.yao import YaoLandmarkDetection
from models.baselines.kim import KimLandmarkDetection


@dataclass
class ModelType:
    name: str
    crop: bool
    resized_images_shape: tuple[int, int]
    resized_points_reference_frame_shape: tuple[int, int]
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    ViT = "ViT"
    ViTLarge = "ViTLarge"
    ViTWithDownscaling = "ViTWithDownscaling"
    ConvNextV2 = "ConvNextV2"
    Yao = "Yao"
    Kim = "Kim"

    @staticmethod
    def model_types():
        return {
            ModelTypes.Kim: ModelType(
                name='Kim',
                crop=False,
                resized_images_shape=(800, 800),#(448, 448),
                resized_points_reference_frame_shape=(256, 256),
                model=KimLandmarkDetection
            ),
            ModelTypes.ViT: ModelType(
                name='ViT',
                crop=False,
                resized_images_shape=(224, 224),
                resized_points_reference_frame_shape=(224, 224),
                model=DirectPointPredictionBasedLandmarkDetection
            ),
            ModelTypes.ViTLarge: ModelType(
                name='ViTLarge',
                crop=False,
                resized_images_shape=(384, 384),
                resized_points_reference_frame_shape=(384, 384),
                model=DirectPointPredictionBasedLandmarkDetection
            ),
            ModelTypes.ViTWithDownscaling: ModelType(
                name='ViTWithDownscaling',
                crop=False,
                resized_images_shape=(450, 450),
                resized_points_reference_frame_shape=(450, 450),
                model=DirectPointPredictionBasedLandmarkDetection
            ),
            ModelTypes.ConvNextV2: ModelType(
                name='ConvNextV2',
                crop=False,
                resized_images_shape=(224, 224),
                resized_points_reference_frame_shape=(224, 224),
                model=DirectPointPredictionBasedLandmarkDetection
            ),
            ModelTypes.Yao: ModelType(
                name='Yao',
                crop=False,
                resized_images_shape=(576, 512),
                resized_points_reference_frame_shape=(576, 512),
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
