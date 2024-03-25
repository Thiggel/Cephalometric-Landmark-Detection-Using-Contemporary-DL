import torch
import torchvision
from matplotlib import pyplot as plt
from dataset.LateralSkullRadiographDataModule import LateralSkullRadiographDataModule
from models.HeatmapBasedLandmarkDetection import HeatmapBasedLandmarkDetection
from models.baselines.chen import fusionVGG19
from models.metrics.MeanRadialError import MeanRadialError

batch_size = 1
resized_image_size = (800, 640)

data_module = LateralSkullRadiographDataModule(
    batch_size=batch_size,
    root_dir='../dataset/benchmark',
    csv_file='points.csv',
    resized_image_size=resized_image_size,
)

original_image_size_mm = data_module.dataset.original_image_size

model = HeatmapBasedLandmarkDetection(
    original_image_size_mm=original_image_size_mm,
    original_image_size=data_module.dataset.original_image_size,
    point_ids=data_module.dataset.point_ids,
    resized_image_size=resized_image_size,
    model=fusionVGG19(
        torchvision.models.vgg19_bn(pretrained=True),
        batch_size,
        data_module.dataset.num_points,
        resized_image_size,
    ),
).cuda()

model.load_state_dict(
    torch.load(
        '../checkpoints/Chen-epoch=56-val_loss=0.15.ckpt',
    )['state_dict'],
)

mre = MeanRadialError(
    resized_image_size=resized_image_size,
    original_image_size_mm=original_image_size_mm,
)

for i in range(20):
    image, targets = next(iter(data_module.train_dataloader()))
    predictions = model(image)
    distance = mre(predictions, targets)

    plt.imshow(image[0].cpu().permute(1, 2, 0))
    plt.scatter(targets[0, :, 0].cpu(), targets[0, :, 1].cpu(), c='r')
    plt.scatter(predictions[0, :, 0].cpu(), predictions[0, :, 1].cpu(), c='b')

    for j in range(targets.shape[1]):
        plt.plot([targets[0, j, 0].cpu(), predictions[0, j, 0].cpu()], [targets[0, j, 1].cpu(), predictions[0, j, 1].cpu()], c='g', linestyle='--')
        distance_text = f'{distance[j].cpu():.2f}'
        plt.text((targets[0, j, 0].cpu() + predictions[0, j, 0].cpu()) / 2, 
                 (targets[0, j, 1].cpu() + predictions[0, j, 1].cpu()) / 2, 
                 distance_text, 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 fontsize=8, color='black')
    
    plt.savefig(f'../test_images/check_distances_{i}.png')
