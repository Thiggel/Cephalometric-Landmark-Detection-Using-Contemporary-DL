import matplotlib.pyplot as plt
import torch
from lightning import Callback


class ImagePredictionLogger(Callback):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def _clamp_points(
        self,
        points: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        image_height, image_width = images.shape[-1], images.shape[-2]

        points = torch.clamp(points, min=0)
        points[..., 0] = torch.clamp(points[..., 0], max=image_width)
        points[..., 1] = torch.clamp(points[..., 1], max=image_height)

        return points

    def on_validation_epoch_start(self, trainer, pl_module):
        images, targets, _, _ = next(iter(trainer.datamodule.val_dataloader()))
        images = images[:self.num_samples]

        preds = pl_module(images)

        preds = self._clamp_points(preds, images)
        targets = self._clamp_points(targets, images)

        images = images.permute(0, 2, 3, 1).cpu().numpy()

        fig, axs = plt.subplots(
            nrows=1,
            ncols=self.num_samples,
            figsize=(60, 100)
        )

        for i, (image, target, pred) in enumerate(zip(images, targets, preds)):
            axs[i].imshow(image, cmap='gray')
            axs[i].scatter(*zip(*target), color='red', s=20)
            axs[i].scatter(*zip(*pred), color='blue', s=20)
            axs[i].axis('off')

        plt.tight_layout()
        plt.savefig('figure.png', bbox_inches='tight')

        image = torch.from_numpy(
            plt.imread('figure.png')
        ).permute(2, 0, 1)

        trainer.logger.experiment.add_image(
            'predictions_vs_targets',
            image,
            global_step=trainer.global_step
        )

        plt.close(fig)
