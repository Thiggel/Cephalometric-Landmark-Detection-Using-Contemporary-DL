from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

from models.DirectPointPredictionBasedLandmarkDetection import \
    DirectPointPredictionBasedLandmarkDetection
from models.ViT import ViT
from models.ConvNextV2 import ConvNextV2
from models.HeatmapBasedLandmarkDetection import HeatmapBasedLandmarkDetection
from models.baselines.yao import YaoGlobalModule, YaoLocalModule
from models.baselines.kim import KimGlobalModule, KimLocalModule


@dataclass
class ModelType:
    resized_image_size: tuple[int, int]
    resized_point_reference_frame_size: tuple[int, int]
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    @staticmethod
    def model_types():
        return {
            'Kim': ModelType(
                resized_image_size=(800, 800),
                resized_point_reference_frame_size=(256, 256),
                model=lambda *args, **kwargs: HeatmapBasedLandmarkDetection(
                    global_module=KimGlobalModule(),
                    local_module=KimLocalModule(),
                    loss=nn.BCEWithLogitsLoss(reduction='none'),
                    *args, **kwargs,
                ),
            ),
            'Yao': ModelType(
                resized_image_size=(576, 512),
                resized_point_reference_frame_size=(576, 512),
                model=lambda *args, **kwargs: HeatmapBasedLandmarkDetection(
                    global_module=YaoGlobalModule(),
                    local_module=YaoLocalModule(),
                    loss=nn.L1Loss(reduction='none'),
                    use_offset_maps=True,
                    *args, **kwargs,
                ),
            ),
            'ViTSmall': ModelType(
                resized_image_size=(224, 224),
                resized_point_reference_frame_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='WinKawaks/vit-small-patch16-224' 
                    ),
                    *args, **kwargs,
                ),
            ),
            'ViTSmallWithDownscaling': ModelType(
                resized_image_size=(902, 902),
                resized_point_reference_frame_size=(902, 902),
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
                resized_point_reference_frame_size=(224, 224),
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
                resized_point_reference_frame_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='google/vit-base-patch16-224-in21k'
                    ),
                    *args, **kwargs,
                ),
            ),
            'ViTLargeImageSize': ModelType(
                resized_image_size=(384, 384),
                resized_point_reference_frame_size=(384, 384),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ViT(
                        model_name='google/vit-base-patch16-384'
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2Small': ModelType(
                resized_image_size=(224, 224),
                resized_point_reference_frame_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-tiny-22k-224',
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2Base': ModelType(
                resized_image_size=(224, 224),
                resized_point_reference_frame_size=(224, 224),
                model=lambda *args, **kwargs: DirectPointPredictionBasedLandmarkDetection(
                    model=ConvNextV2(
                        model_name='facebook/convnextv2-base-22k-224',
                    ),
                    *args, **kwargs,
                ),
            ),
            'ConvNextV2LargeImageSize': ModelType(
                resized_image_size=(384, 384),
                resized_point_reference_frame_size=(384, 384),
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
