import torch
from torch import nn
from typing import Callable
from transformers import ConvNextV2Model

from models.ViT import Downscaling


class ConvNextV2(nn.Module):
    def __init__(
        self,
        model_type: str = 'tiny',
        output_size: int = 44,
    ):
        super().__init__()

        self.downscaling = Downscaling()

        self.model = self._load_model(model_type)

        self.head = nn.Linear(
            self.model.config.hidden_sizes[-1],
            2 * output_size
        )

        self.output_size = output_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.downscaling(images)
        output = self.model(images).pooler_output
        return self.head(output).reshape(-1, self.output_size, 2)

    def _load_model(self, model_type: str) -> Callable:
        models = {
            'tiny': 'facebook/convnextv2-atto-1k-224',
            'normal': 'facebook/convnextv2-huge-1k-224',
        }

        if model_type not in models:
            raise ValueError(
                f"model_type must be one of {list(models.keys())}"
            )

        model = ConvNextV2Model.from_pretrained(models[model_type])

        return model
