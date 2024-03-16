import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation


class Segformer(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_size: int,
        *args,
        **kwargs
    ):
        super().__init__()

        self.backbone, self.config = self._load_model(model_name)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1,8, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3 * output_size),
        )

        self.output_size = output_size

    def _load_model(self, model_name: str) -> nn.Module:
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        return model, model.config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        output = self.head(output)

        return output
