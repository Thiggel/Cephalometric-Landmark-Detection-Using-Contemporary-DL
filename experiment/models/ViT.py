import torch
from torch import nn
from transformers import ViTModel, ViTConfig


class Downscaling(nn.Sequential):
    def __init__(self):
        super().__init__(
            # input shape: 450, 450, 1
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # output shape: 224, 224, 3
        )


class ViT(nn.Module):
    def __init__(self, model_type: str = 'tiny', output_size: int = 44):
        super().__init__()

        self.downscaling = Downscaling()
        self.model, self.config = self._load_model(model_type)
        self.head = nn.Linear(self.config.hidden_size, 2 * output_size)
        self.output_size = output_size

    def _load_model(self, model_type: str) -> nn.Module:
        if model_type == 'tiny':
            model_name = 'WinKawaks/vit-tiny-patch16-224'
        elif model_type == 'normal':
            model_name = 'google/vit-base-patch16-224-in21k'
        else:
            raise ValueError("model_type must be either 'tiny' or 'normal'")

        config = ViTConfig.from_pretrained(model_name)

        model = ViTModel(config)

        return model, config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.downscaling(images)
        output = self.model(images).last_hidden_state[:, 0, :]
        output = self.head(output).reshape(-1, self.output_size, 2)

        return output
