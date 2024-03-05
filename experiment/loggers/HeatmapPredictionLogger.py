import matplotlib.pyplot as plt
import torch
from lightning import Callback
from lightning import Trainer, LightningModule

from utils.clamp_points import clamp_points
from utils.resize_points import resize_points


class HeatmapPredictionLogger(Callback):
    def __init__(
        self,
        num_samples: int,
        module_name: str
    ):
        super().__init__()
        self.num_samples = num_samples
        self.module_name = module_name

    def show_heatmap(
        self,
        axs: plt.Axes,
        heatmap: torch.Tensor,
        title: str,
        target: torch.Tensor,
        pred: torch.Tensor,
        show_target: bool = True
    ) -> None:
        axs.imshow(heatmap, cmap='hot')
        axs.set_title(title)
        if show_target:
            axs.scatter(*zip(*target), color='red', s=20, label='Targets')
        axs.scatter(*zip(*pred), color='blue', s=20, label='Predictions')
        axs.legend()

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        batch = next(iter(
            trainer.datamodule.val_dataloader()
        ))

        images, targets = batch

        images = images[:self.num_samples]
        targets = targets[:self.num_samples]
        (
            _,
            preds,
            global_heatmaps,
            local_heatmaps,
            target_heatmaps,
        ) = pl_module.step((images, targets))

        preds = clamp_points(preds, images).cpu().numpy()
        targets = clamp_points(targets, images).cpu().numpy()
        global_heatmaps = global_heatmaps.sum(dim=1).cpu().numpy()
        local_heatmaps = local_heatmaps.sum(dim=1).cpu().numpy()
        target_heatmaps = target_heatmaps.sum(dim=1).cpu().numpy()

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        fig, axs = plt.subplots(
            nrows=3,
            ncols=self.num_samples,
            figsize=(60, 100)
        )

        for i, (
            target,
            pred,
            global_heatmap,
            local_heatmap,
            target_heatmap
        ) in enumerate(
            zip(
                targets,
                preds,
                global_heatmaps,
                local_heatmaps,
                target_heatmaps
            )
        ):
            self.show_heatmap(
                axs[0, i],
                global_heatmap,
                'Global Heatmap',
                target,
                pred
            )

            self.show_heatmap(
                axs[1, i],
                local_heatmap,
                'Local Heatmap',
                target,
                pred
            )

            self.show_heatmap(
                axs[2, i],
                target_heatmap,
                'Target Heatmap',
                target,
                pred,
                show_target=False
            )

        plt.tight_layout()
        if not os.path.exists('figures'):
            os.makedirs('figures')

        plt.savefig(f'figures/figure_heatmaps_{self.module_name}.png', bbox_inches='tight')

        image = torch.from_numpy(
            plt.imread('figures/figure_heatmaps.png')
        ).permute(2, 0, 1)

        trainer.logger.experiment.add_image(
            'heatmaps_vs_targets',
            image,
            global_step=1
        )

        plt.close(fig)
