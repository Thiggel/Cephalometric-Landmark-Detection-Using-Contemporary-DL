import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
import lightning as L

from utils.CanExtractPatches import CanExtractPatches
from models.losses.MaskedWingLoss import MaskedWingLoss


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


class DetectionModule:
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


class GlobalDetectionModule(nn.Module, DetectionModule):
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

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.upsample(x)
        coordinates = self.get_highest_points(x)

        return coordinates, x


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

        return self.backbone[4](x)


class LocalCorrectionModule(nn.Module, DetectionModule):
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

        highest_points = self.get_highest_points(x)

        return highest_points, x


class LandmarkDetection(nn.Module, CanExtractPatches):
    def __init__(
        self,
        patch_size: tuple[int, int] = (96, 96)
    ):
        super(LandmarkDetection, self).__init__()

        self.global_module = GlobalDetectionModule()
        self.local_module = LocalCorrectionModule()

        self.patch_size = patch_size

        self.middle_of_patch = torch.tensor([
            patch_size[0] // 2,
            patch_size[1] // 2
        ]).unsqueeze(0).unsqueeze(0)

    def refine_point_predictions(
        self,
        points_full: torch.Tensor,
        points_patch: torch.Tensor,
    ) -> torch.Tensor:

        refinement = points_patch - self.middle_of_patch

        return points_full + refinement

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        points_full, heatmaps_full = self.global_module(images)

        patches = self._extract_patches(images, points_full) \
            .repeat(1, 1, 3, 1, 1)

        points_patch = []
        heatmaps_patch = []

        for image_patches in patches:
            points, heatmap = self.local_module(image_patches)

            points_patch.append(points.squeeze())
            heatmaps_patch.append(heatmap)

        points_patch = torch.stack(points_patch)
        heatmaps_patch = torch.stack(heatmaps_patch)

        points_refined = self.refine_point_predictions(
            points_full,
            points_patch
        )

        return heatmaps_full, heatmaps_patch, points_refined


class YaoLandmarkDetection(
    L.LightningModule,
    DetectionModule,
    CanExtractPatches
):
    def __init__(
        self,
        point_ids: list[str] = [],
        reduce_lr_patience: int = 25,
        model_size: str = 'tiny',
        gaussian_sigma: int = 1,
        gaussian_alpha: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_size = model_size
        self.reduce_lr_patience = reduce_lr_patience
        self.model = LandmarkDetection()
        self.point_ids = point_ids

        self.gaussian_sigma = gaussian_sigma
        self.gaussian_alpha = gaussian_alpha
        self.patch_size = (96, 96)
        self.loss = MaskedWingLoss()

    def forward(self, x):
        return self.model(x)[-1]

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ):
        images, points, target_heatmaps_full, target_heatmaps_patch = batch

        (
            heatmaps_full,
            heatmaps_patch,
            point_predictions
        ) = self.model(images)

        loss = F.l1_loss(heatmaps_full, target_heatmaps_full) + \
            F.l1_loss(heatmaps_patch, target_heatmaps_patch)

        return loss, point_predictions

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _ = self.step(batch)

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, predictions = self.step(batch)
        targets = batch[1]

        _, unreduced_mm_error = self.loss(
            predictions,
            targets,
            with_mm_error=True
        )

        return loss, unreduced_mm_error

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, unreduced_mm_error = self.validation_test_step(batch)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log(
            'val_mm_error',
            unreduced_mm_error.mean(),
            prog_bar=True,
            on_epoch=True
        )

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, unreduced_mm_error = self.validation_test_step(batch)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', unreduced_mm_error[id].mean())

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', unreduced_mm_error.mean(), prog_bar=True)

        return loss

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
