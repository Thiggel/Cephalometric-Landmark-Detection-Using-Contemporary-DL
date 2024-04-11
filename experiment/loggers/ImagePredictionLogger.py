import matplotlib.pyplot as plt
import torch
from lightning import Callback, Trainer, LightningModule
import os


class ImagePredictionLogger(Callback):
    def __init__(
        self,
        num_samples: int,
        resized_image_size: tuple[int, int],
        model_name: str,
        dataset_name: str,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.resized_image_size = resized_image_size
        self.module_name = model_name
        self.dataset_name = dataset_name

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        val_loader = trainer.datamodule.val_dataloader()
        val_loader_iter = iter(val_loader)
        original_batch_size = val_loader.batch_size

        if original_batch_size == 1:
            images, targets = [], []
            for _ in range(self.num_samples):
                image, target = next(val_loader_iter)
                images.append(image)
                targets.append(target)

            images = torch.cat(images, dim=0)
            targets = torch.cat(targets, dim=0)
        else:
            images, targets = next(val_loader_iter)

        images = images[:self.num_samples]

        pl_module.show_images(images, targets)

        plt.tight_layout()

        if not os.path.exists('figures'):
            os.makedirs('figures')

        path = f'figures/figure_{self.module_name}.png'
        plt.savefig(path, bbox_inches='tight')

        try:
            image = torch.from_numpy(
                plt.imread(path)
            ).permute(2, 0, 1)

            trainer.logger.experiment.add_image(
                'predictions_vs_targets',
                image,
                global_step=1
            )
        except Exception as e:
            print(f'Error: {e}')

        plt.close()
