import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
import lightning as L


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


class CanExtractPatches:
    def _extract_patch(
        self,
        image: torch.Tensor,
        x: int,
        y: int
    ) -> torch.Tensor:
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

        return patches.unsqueeze(2).repeat(1, 1, 3, 1, 1)


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

        patches = self._extract_patches(images, points_full)

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


class YaoLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        original_image_size: tuple[int, int],
        point_ids: list[str] = [],
        reduce_lr_patience: int = 25,
        model_size: str = 'tiny',
        gaussian_sigma: int = 20,
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

        self.original_image_size = original_image_size

    def forward(self, x):
        return self.model(x)[-1]

    def masked_l2_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        l2_loss = ((predictions - targets) ** 2).sum(-1).sqrt()

        mask = (targets > 0).prod(-1)
        masked_l2_loss = l2_loss * mask

        return masked_l2_loss

    def get_mm_error(
        self,
        non_reduced_loss: torch.Tensor,
    ) -> torch.Tensor:
        px_per_m = torch.tensor(7_756)
        m_to_mm = 1000

        return (
            non_reduced_loss
            / px_per_m.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            * m_to_mm
        ).squeeze().mean(0)

    def _resize_points(
        self,
        points: torch.Tensor,
        image_size: tuple[int, int]
    ) -> torch.Tensor:
        original_width, original_height = self.original_image_size
        width, height = image_size
        width_resize_factor = width / original_width
        height_resize_factor = height / original_height

        points[..., 0] *= width_resize_factor
        points[..., 1] *= height_resize_factor

        return points

    def _generate_heatmap(
        self,
        target_points: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_points, _ = target_points.size()
        image_size = images.shape[-2:]
        width, height = image_size

        unresized_target_points = target_points.clone()
        target_points = self._resize_points(target_points, image_size)

        x = torch.arange(0, width).float()
        y = torch.arange(0, height).float()
        xx, yy = torch.meshgrid(x, y)

        print(target_points.shape)
        print(target_points)

        heatmap_list = []
        for batch_idx in range(batch_size):
            heatmap = torch.zeros(width, height)
            for point_idx in range(num_points):
                x_center, y_center = target_points[batch_idx, point_idx]

                if (
                    x_center <= 0 or x_center >= width or
                    y_center <= 0 or y_center >= height
                ):
                    continue

                heatmap += torch.exp(
                    -0.5 * (
                        (xx - x_center) ** 2 +
                        (yy - y_center) ** 2
                    ) / self.gaussian_sigma ** 2
                )

            heatmap /= num_points

            heatmap *= self.gaussian_alpha

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(heatmap.cpu().detach().numpy())
            ax[1].imshow(images[batch_idx].squeeze().cpu().detach().numpy())
            ax[1].scatter(target_points[batch_idx, :, 0], target_points[batch_idx, :, 1])

            from torchvision.transforms import Resize

            resized_image = Resize(self.original_image_size[::-1])(images[batch_idx].unsqueeze(0))

            ax[2].imshow(resized_image.squeeze().cpu().detach().numpy())
            ax[2].scatter(unresized_target_points[batch_idx, :, 0], unresized_target_points[batch_idx, :, 1])
            plt.show()
            exit()

            heatmap_list.append(heatmap.unsqueeze(0))

        return torch.stack(heatmap_list, dim=0)

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        images, points = batch

        (
            heatmaps_full,
            heatmaps_patch,
            point_predictions
        ) = self.model(images)

        target_heatmaps = self._generate_heatmap(
            points,
            images
        )

        #non_reduced_loss = self.masked_l2_loss(predictions, points)

        #mm_error = self.get_mm_error(
        #    non_reduced_loss
        #) if with_mm_error else None

        #loss = non_reduced_loss.mean()

        #return loss, mm_error

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

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error = self.step(batch, with_mm_error=True)

        mm_error = mm_error.mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, mm_error = self.step(batch, with_mm_error=True)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean())

        mm_error = mm_error.mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error, prog_bar=True)

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
    
