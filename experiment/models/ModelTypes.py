from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
import torchvision

from models.DirectPointPredictionBasedLandmarkDetection import \
    DirectPointPredictionBasedLandmarkDetection
from models.HeatmapBasedLandmarkDetection import \
    HeatmapBasedLandmarkDetection
from models.baselines.chen import fusionVGG19, ChenConvNext
from models.backbones.ViT import ViT
from models.backbones.ConvNextV2 import ConvNextV2
from models.backbones.Segformer import Segformer


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
                model=lambda \
                    batch_size, \
                    output_size, \
                    resized_image_size, \
                    *args, \
                    **kwargs \
                : HeatmapBasedLandmarkDetection(
                    model=fusionVGG19(
                        torchvision.models.vgg19_bn(pretrained=True),
                        batch_size,
                        output_size,
                        resized_image_size,
                    ),
                    *args, **kwargs,
                ),
            ),
            'ChenConvNext': ModelType(
                resized_image_size=(800, 640),
                model=lambda \
                    batch_size, \
                    output_size, \
                    resized_image_size, \
                    *args, \
                    **kwargs \
                : HeatmapBasedLandmarkDetection(
                    model=ChenConvNext(
                        ConvNextV2(
                            model_name='facebook/convnextv2-base-22k-224',
                        ),
                        batch_size,
                        output_size,
                        resized_image_size,
                    ),
                    *args, **kwargs,
                ),
            ),
            'SegformerSmall': ModelType(
                resized_image_size=(512, 512),
                model=lambda output_size, *args, **kwargs: HeatmapBasedLandmarkDetection(
                    model=Segformer(
                        model_name='nvidia/segformer-b4-finetuned-ade-512-512',
                        output_size=output_size,
                    ),
                    *args, **kwargs,
                ),
            ),
            'SegformerLarge': ModelType(
                resized_image_size=(640, 640),
                model=lambda output_size, *args, **kwargs: HeatmapBasedLandmarkDetection(
                    model=Segformer(
                        model_name='nvidia/segformer-b5-finetuned-ade-640-640',
                        output_size=output_size,
                    ),
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
                model=lambda output_size, *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        output_size=output_size,
                        model_name='google/vit-base-patch16-384'
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextSmall': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-tiny-22k-224',
                        output_size=output_size,
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextBase': ModelType(
                resized_image_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-base-22k-224',
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextLargeImageSize': ModelType(
                resized_image_size=(384, 384),
                model=lambda output_size, *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        output_size=output_size,
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
