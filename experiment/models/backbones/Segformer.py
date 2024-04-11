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

        self.backbone.decode_head.classifier = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels=self.config.decoder_hidden_size,
                out_channels=output_size * 3,
                kernel_size=1,
            ),
        )

        self.output_size = output_size

    def _load_model(self, model_name: str) -> nn.Module:
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        return model, model.config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(images).logits

        return output
