import torch
import torch.nn as nn
import torchvision.models as models

from models.baselines.aspp.ASPP import ASPP


class GlobalResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._init_backbone()

        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 64, 1),
        ])

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

    def _init_backbone(self):
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        backbone = nn.Sequential(*modules)

        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone[0:4](x)

        layer1 = self.backbone[4](x)
        layer2 = self.backbone[5](layer1)
        layer3 = self.backbone[6](layer2)
        layer4 = self.backbone[7](layer3)

        upsample1 = self.upsample(
            self.decoder_layers[0](layer4)
        ) + layer3

        upsample2 = self.upsample(
            self.decoder_layers[1](upsample1)
        ) + layer2

        upsample3 = self.upsample(
            self.decoder_layers[2](upsample2)
        ) + layer1

        return upsample3


class YaoGlobalModule(nn.Module):
    def __init__(
        self,
        output_size: int = 44
    ):
        super().__init__()

        self.output_size = output_size

        self.backbone = GlobalResNetBackbone()

        self.aspp = ASPP(64, 256, [1, 6, 12, 18])

        # 3 * output_size because we are predicting 44 heatmaps
        # and 44 offset maps for x and y
        self.conv = nn.Conv2d(256, 3 * output_size, 1)

        self.upsample = nn.Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.upsample(x)

        return x


class LocalResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._init_backbone()

        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 64, 1),
        ])

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

    def _init_backbone(self):
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        backbone = nn.Sequential(*modules)

        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone[0:4](x.repeat(1, 3, 1, 1))

        return self.backbone[4](x)


class YaoLocalModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = LocalResNetBackbone()

        self.aspp = ASPP(64, 256, [1, 6, 12, 18])

        self.conv = nn.Conv2d(256, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

        self.output_size = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x = self.aspp(features)
        x = self.conv(x)
        x = self.upsample(x)

        return x
