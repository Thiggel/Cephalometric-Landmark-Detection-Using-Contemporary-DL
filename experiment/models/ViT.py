import torch
from torch import nn
from transformers import ViTModel


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
    def __init__(
        self,
        model_name: str = 'tiny',
        output_size: int = 44,
        downscale: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.downscale = downscale
        self.downscaling = Downscaling()
        self.model, self.config = self._load_model(model_name)
        self.head = nn.Linear(self.config.hidden_size, 2 * output_size)
        self.output_size = output_size

    def _load_model(self, model_name: str) -> nn.Module:
        if model_type == 'tiny':
            model_name = 'WinKawaks/vit-small-patch16-224' # 'WinKawaks/vit-tiny-patch16-224'
        elif model_type == 'normal':
            model_name = 'google/vit-base-patch16-224-in21k'
        elif model_type == 'large':
            model_name = 'google/vit-large-patch16-384'
        else:
            raise ValueError("model_type must be either 'tiny' or 'normal'")

        model = ViTModel.from_pretrained(model_name)

        return model, model.config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.downscale:
            images = self.downscaling(images)
        else:
            images = images.repeat(1, 3, 1, 1)

        output = self.model(images).last_hidden_state[:, 0, :]
        output = self.head(output).reshape(-1, self.output_size, 2)

        return output
