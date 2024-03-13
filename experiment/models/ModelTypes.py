from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

from models.DirectPointPredictionBasedLandmarkDetection import \
    DirectPointPredictionBasedLandmarkDetection
from models.baselines.chen import ChenLandmarkPrediction
from models.ViT import ViT
from models.ConvNextV2 import ConvNextV2


@dataclass
class ModelType:
    resized_image_size: tuple[int, int]
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    @staticmethod
    def model_types():
        return {
            'Chen': ModelType(
                resized_image_size=(800, 640),
                model=lambda *args, **kwargs: ChenLandmarkPrediction(
                    *args, **kwargs,
                ),
            ),
            'ViTSmall': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='WinKawaks/vit-small-patch16-224',
                        output_size=output_size,
                    ),
                    *args, **kwargs,
                ),
            ),
            'ViTSmallWithDownscaling': ModelType(
                resized_image_size=(902, 902),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='WinKawaks/vit-small-patch16-224',
                        downscale=True,
                    ),
                    *args, **kwargs,
                ),
            ),
            'ViTSmallWithComplexMLPHead': ModelType(
                resized_image_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='WinKawaks/vit-small-patch16-224',
                        complex_mlp_head=True,
                    ),
                    *args, **kwargs,
                ),

            ),
            'ViTBase': ModelType(
                resized_image_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='google/vit-base-patch16-224-in21k'
                    ),
                    *args, **kwargs,
                ),
            ),
            'ViTLargeImageSize': ModelType(
                resized_image_size=(384, 384),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='google/vit-base-patch16-384'
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2Small': ModelType(
                resized_image_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-tiny-22k-224',
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2Base': ModelType(
                resized_image_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-base-22k-224',
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2LargeImageSize': ModelType(
                resized_image_size=(384, 384),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-tiny-22k-384',
                    ),
                    *args, **kwargs,
                ),
            ),
        }

    @staticmethod
    def get_model_type(name: str) -> ModelType:
        return ModelTypes.model_types()[name]

    @staticmethod
    def get_model_types() -> list[str]:
        return list(map(
            lambda x: x,
            list(ModelTypes.model_types().keys())
        ))
