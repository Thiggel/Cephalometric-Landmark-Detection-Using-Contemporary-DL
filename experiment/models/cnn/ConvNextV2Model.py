import os
import torch
from torch import nn
from typing import Callable

from ..ViT import Downscaling
from .ConvNextV2.convnextv2 import \
    ConvNeXtV2, \
    convnextv2_atto, \
    convnextv2_huge


class ConvNextV2Model(nn.Module):
    def __init__(
        self,
        model_type: str = 'tiny',
        output_size: int = 44,
    ):
        super().__init__()

        self.downscaling = Downscaling()

        self.model = self._load_model(model_type)

        self.head = nn.Linear(self.config.hidden_size, 2 * output_size)

        self.output_size = output_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.downscaling(images)
        output = self.model(images)
        output = self.head(output).reshape(-1, self.output_size, 2)

    @property
    def _models(self) -> dict[str, Callable[[], ConvNeXtV2]]:
        return {
            'tiny': lambda:  convnextv2_atto(),
            'normal': lambda: convnextv2_huge(),
        }

    def _checkpoints(self) -> dict[str, str]:
        return {
            'tiny': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
            'normal': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt',
        }

    def _load_checkpoint(self, model_type: str, model: ConvNeXtV2) -> None:
        checkpoint = self._checkpoints()[model_type]

        path = os.path.join(
            os.getcwd(),
            'checkpoints/',
            os.path.basename(checkpoint)
        )

        torch.hub.download_url_to_file(
            checkpoint,
            path
        )

        model.load_state_dict(torch.load(path))

    def _load_model(self, model_type: str) -> Callable:
        if model_type not in self._models:
            raise ValueError(
                f"model_type must be one of {list(self._models.keys())}"
            )

        model = self._models[model_type]()

        self._load_checkpoint(model_type, model)
