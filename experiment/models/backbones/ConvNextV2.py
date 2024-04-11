import torch
from torch import nn
from typing import Callable
from transformers import ConvNextV2Model


class ConvNextV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_size: int = 44,
        *args,
        **kwargs
    ):
        super().__init__()

        self.model = self._load_model(model_name)

        self.head = nn.Linear(
            self.model.config.hidden_sizes[-1],
            2 * output_size
        )

        self.output_size = output_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images).pooler_output
        return self.head(output).reshape(-1, self.output_size, 2)

    def _load_model(self, model_name: str) -> Callable:
        model = ConvNextV2Model.from_pretrained(model_name)

        return model
