import os
import torch
from torch import nn

from RevCol.revcol import (
    FullNet,
    revcol_tiny,
    revcol_small,
    revcol_base,
    revcol_large,
    revcol_xlarge
)


class RevCol(nn.Module):
    def __init__(
        self,
        model_name: str = 'revol_large',
        num_channels: int = 1
    ):
        super().__init__()

        self.model = self._get_model(model_name)

        print(self.model)

    @property
    def _model_dict(self) -> dict:
        return {
            'revol_tiny': {
                'model': revcol_tiny,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_tiny_1k.pth',
            },
            'revol_small': {
                'model': revcol_small,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_small_1k.pth',
            },
            'revol_base': {
                'model': revcol_base,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_base_1k.pth',
            },
            'revol_large': {
                'model': revcol_large,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_large_22k.pth',
            },
            'revol_xlarge': {
                'model': revcol_xlarge,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_large_22k.pth',
            },
        }

    def _load_model_params(self, model: FullNet, model_name) -> None:
        path = os.path.join(
            os.getcwd(),
            'RevCol',
            f'{model_name}.pth'
        )

        if not os.path.exists(path):
            torch.hub.download_url_to_file(
                self._model_dict[model_name]['url'],
                path
            )

        print(torch.load(path))

        model.load_state_dict(torch.load(path))

    def _get_model(self, model_name: str) -> FullNet:
        if model_name not in self._model_dict:
            print("Model not found!")
            return

        model = self._model_dict[model_name]['model'](save_memory=False)
        print(model)
        exit()

        self._load_model_params(model, model_name)

        self._adjust_model(model)

        return model

    def _adjust_model(self, model: FullNet) -> FullNet:
        model.stem = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                model.channels[0],
                kernel_size=4,
                stride=4
            ),
            nn.LayerNorm(
                model.channels[0],
                eps=1e-6,
                data_format="channels_first"
            )
        )

    def forward(self, x):
        return x

mod = RevCol(model_name='revol_tiny', num_channels=1)
