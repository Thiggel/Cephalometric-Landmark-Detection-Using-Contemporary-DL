import os
import torch
from torch import nn

from .RevCol.revcol import (
    FullNet,
    revcol_tiny,
    revcol_small,
    revcol_base,
    revcol_large,
    revcol_xlarge
)

from ..ViT import Downscaling


class RevColModel(nn.Module):
    def __init__(
        self,
        model_type: str = 'tiny'
    ):
        super().__init__()

        self.downscaling = Downscaling()
        self.model = self._get_model(model_type)

    @property
    def _model_dict(self) -> dict:
        return {
            'tiny': {
                'model': revcol_tiny,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_tiny_1k.pth',
            },
            'normal': {
                'model': revcol_large,
                'url': 'https://github.com/megvii-research/RevCol/releases/download/checkpoint/revcol_large_22k.pth',
            },
        }

    def _load_model_params(self, model: FullNet, model_type) -> None:
        path = os.path.join(
            os.getcwd(),
            'checkpoints',
            f'{model_type}.pth'
        )

        if not os.path.exists(path):
            torch.hub.download_url_to_file(
                self._model_dict[model_type]['url'],
                path
            )

        state_dict = torch.load(path)['model']

        model.load_state_dict(state_dict, strict=False)

    def _get_model(self, model_type: str) -> FullNet:
        if model_type not in self._model_dict:
            print("Model not found!")
            return

        model = self._model_dict[model_type]['model'](save_memory=False)

        self._load_model_params(model, model_type)

        return model

    def forward(self, x):
        downscaled = self.downscaling(x)
        output = self.model(downscaled)
        
        print(len(output))

        exit()
