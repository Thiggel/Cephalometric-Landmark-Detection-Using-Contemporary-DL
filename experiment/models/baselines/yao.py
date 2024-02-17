import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: int):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d)
            for d_idx, d in enumerate(dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum([block(x) for block in self.aspp_blocks])


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


class GlobalDetectionModule(nn.Module):
    def __init__(
        self,
        output_size: int = 44
    ):
        super(GlobalDetectionModule, self).__init__()

        self.output_size = output_size

        self.backbone = GlobalResNetBackbone()

        self.aspp = ASPP(64, 256, [1, 6, 12, 18])

        self.conv = nn.Conv2d(256, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

    def get_highest_points(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size = feature_map.shape[0]
        values, indices = feature_map.view(batch_size, -1).topk(
            self.output_size
        )

        indices = indices % (feature_map.shape[2] * feature_map.shape[3])

        y_coords, x_coords = (
            indices // feature_map.shape[3],
            indices % feature_map.shape[3]
        )

        return torch.stack([x_coords, y_coords], dim=2)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.upsample(x)
        coordinates = self.get_highest_points(x)
        return coordinates


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
        x = self.backbone[0:4](x)

        layer1 = self.backbone[4](x)
        print(layer1.shape)
        exit()
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


class LocalCorrectionModule(nn.Module):
    def __init__(self):
        super(LocalCorrectionModule, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.aspp = ASPP(512, 256, [1, 6, 12, 18])
        self.upsample = nn.Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x = self.aspp(features)
        return self.upsample(x)


class LandmarkDetection(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int] = (96, 96)
    ):
        super(LandmarkDetection, self).__init__()
        self.global_module = GlobalDetectionModule()
        self.local_module = LocalCorrectionModule()
        self.patch_size = patch_size

    def _extract_patch(self, image, x, y):
        image_width, image_height = image.shape[-2:]
        patch_width, patch_height = self.patch_size

        x = min(image_width, max(0, x))
        y = min(image_height, max(0, y))

        x_offset = patch_width // 2
        y_offset = patch_height // 2

        x_min = max(0, x - x_offset)
        x_max = min(image_width, x + x_offset)
        y_min = max(0, y - y_offset)
        y_max = min(image_height, y + y_offset)

        x_offset = max(0, x_offset - x)
        y_offset = max(0, y_offset - y)

        patch = torch.zeros(patch_width, patch_height)

        patch[
            x_offset:x_offset + x_max - x_min,
            y_offset:y_offset + y_max - y_min,
        ] = image[
            ...,
            x_min:x_max,
            y_min:y_max,
        ]

        return patch

    def _extract_patches(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = coords.shape

        patches = torch.zeros(
            batch_size,
            num_points,
            *self.patch_size
        )

        for i in range(batch_size):
            for j in range(num_points):
                x, y = coords[i, j]
                patch = self._extract_patch(images[i], x, y)
                patches[i, j] = patch

        return patches

    def forward(self, x):
        coordinates = self.global_module(x)

        patches = self._extract_patches(x, coordinates)
        points = torch.stack([
            self.local_module(patch).unsqueeze(0) for patch in patches
        ], dim=1)

        return points
