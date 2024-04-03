from __future__ import print_function, division
import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils.clamp_points import clamp_points
import numpy as np

from models.losses.HeatmapOffsetmapLoss import HeatmapOffsetmapLoss
from models.metrics.MeanRadialError import MeanRadialError
from utils.HeatmapHelper import HeatmapHelper


class HeatmapBasedLandmarkDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        point_ids: list[str],
        original_image_size: tuple[float, float],
        original_image_size_mm: tuple[float, float],
        resized_image_size: int = (800, 640),
        batch_size: int = 1,
        output_size: int = 19,
        reduce_lr_patience: int = 25,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.batch_size = batch_size

        self.loss = HeatmapOffsetmapLoss(resized_image_size)

        self.mean_radial_error = MeanRadialError(
            resized_image_size=resized_image_size,
            original_image_size_mm=original_image_size_mm,
        )

        self.point_ids = point_ids

        self.reduce_lr_patience = reduce_lr_patience

        self.heatmap_helper = HeatmapHelper(
            original_image_size=original_image_size,
            resized_image_size=resized_image_size,
        )

    def forward_with_heatmaps(self, x):
        output = self.model(x)
        return output

    def plot_image(
        self,
        image: torch.Tensor,
        num_samples: int,
        axs: plt.Axes,
        targets: torch.Tensor,
        preds: torch.Tensor,
    ) -> None:
        axs.imshow(image, cmap='gray')
        axs.scatter(*zip(*targets), color='red', s=20)
        axs.scatter(*zip(*preds), color='blue', s=20)
        axs.axis('off')

    def get_target_heatmaps_for_visualization(
        self,
        targets: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return self.loss.cut_out_rectangles(
            self.loss.general_heatmap,
            targets,
            height,
            width,
        ).max(dim=1).values

    def show_images(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        feature_maps = (
            torch.cat([
                self.forward_with_heatmaps(image.unsqueeze(0))
                for image in images
            ], dim=0) if self.batch_size == 1
            else self.forward_with_heatmaps(images)
        )

        preds = self.get_points(feature_maps)

        num_points = preds.size(1)
        heatmaps = feature_maps[:, :num_points].max(dim=1).values
        _, height, width = heatmaps.size()

        target_heatmaps = self.get_target_heatmaps_for_visualization(
            targets, height, width
        )

        preds = clamp_points(preds, images).cpu().numpy()
        targets = clamp_points(targets, images).cpu().numpy()

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        num_samples = images.shape[0]

        colors = [(1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 0, 1)]
        heatmap_cmap = LinearSegmentedColormap.from_list("heatmap", colors)

        fig, axs = plt.subplots(2, num_samples, figsize=(20, num_samples * 20))

        for idx, (image, heatmap, target_heatmap, target, pred) in enumerate(
            zip(images, heatmaps, target_heatmaps, targets, preds)
        ):
            first_plot = axs[0, idx] if num_samples > 1 else axs[0]
            self.plot_image(image, num_samples, first_plot, target, pred)
            first_plot.imshow(
                heatmap.detach().cpu(), alpha=0.5, cmap=heatmap_cmap,
            )

            second_plot = axs[1, idx] if num_samples > 1 else axs[1]
            self.plot_image(image, num_samples, second_plot, target, pred)
            second_plot.imshow(
                target_heatmap.detach().cpu(), alpha=0.5, cmap=heatmap_cmap,
            )

    def forward(self, x):
        output = self.forward_with_heatmaps(x)

        return self.get_points(output)

    def regression_voting(self, heatmaps, R):
        topN = int(R * R * 3.1415926)
        heatmap = heatmaps
        imageNum, featureNum, h, w = heatmap.size()
        landmarkNum = int(featureNum / 3)
        heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)

        predicted_landmarks = torch.zeros((imageNum, landmarkNum, 2), device=self.device)
        Pmap = heatmap[:, 0:landmarkNum, :].data
        Xmap = torch.round(heatmap[:, landmarkNum : landmarkNum * 2, :].data * R).long() * w
        Ymap = torch.round(heatmap[:, landmarkNum * 2 : landmarkNum * 3, :].data * R).long()
        topkP, indexs = torch.topk(Pmap, topN)

        for imageId in range(imageNum):
            for landmarkId in range(landmarkNum):

                topnXoff = Xmap[imageId][landmarkId][
                    indexs[imageId][landmarkId]
                ]  # offset in x direction

                topnYoff = Ymap[imageId][landmarkId][
                    indexs[imageId][landmarkId]
                ]  # offset in y direction

                VotePosi = (
                    (topnXoff + topnYoff + indexs[imageId][landmarkId])
                    .cpu()
                    .numpy()
                    .astype("int")
                )

                tem = VotePosi[VotePosi >= 0]
                maxid = 0

                if len(tem) > 0:
                    maxid = np.argmax(np.bincount(tem))

                x = maxid // w
                y = maxid - x * w
                #x, y = x / (h - 1), y / (w - 1)
                predicted_landmarks[imageId][landmarkId] = torch.tensor([y, x], device=self.device)
        return predicted_landmarks

    def get_points(self, model_output: torch.Tensor):
        return self.regression_voting(model_output, 40)

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        with_mm_error: bool = False
    ):
        inputs, targets = batch

        predictions = self.forward_with_heatmaps(inputs)

        loss = self.loss(
            predictions,
            targets,
        )

        point_predictions = self.get_points(predictions)

        unreduced_mm_error = self.mean_radial_error(
            point_predictions,
            targets,
        ) if with_mm_error else None

        return loss, unreduced_mm_error, point_predictions, targets

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        loss, _, _, _ = self.step(batch)

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
        loss, mm_error, _, _ = self.step(batch, with_mm_error=True)

        mm_error = mm_error.mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mm_error', mm_error, prog_bar=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ):
        (
            loss,
            mm_error,
            predictions,
            targets
        ) = self.step(batch, with_mm_error=True)

        for (id, point_id) in enumerate(self.point_ids):
            self.log(f'{point_id}_mm_error', mm_error[id].mean())

        mm_error = mm_error.mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mm_error', mm_error, prog_bar=True)

        self.log(
            'percent_under_1mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 1)
        )
        self.log(
            'percent_under_2mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 2)
        )
        self.log(
            'percent_under_3mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 3)
        )
        self.log(
            'percent_under_4mm',
            self.mean_radial_error.percent_under_n_mm(mm_error, targets, 4)
        )

        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1.0
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    patience=self.reduce_lr_patience
                ),
                'monitor': 'val_loss'
            },
        }

