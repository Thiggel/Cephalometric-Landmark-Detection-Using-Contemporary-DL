from __future__ import print_function, division
import torch.optim as optim
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataLoader import Rescale, RandomCrop, ToTensor, LandmarksDataset
import models
import train
import lossFunction
import argparse
from tqdm import tqdm
from LateralSkullRadiographDataModule import LateralSkullRadiographDataModule
from ViT import ViT

plt.ion()  # interactive mode
def tuple_type(arg):
    try:
        return list(map(int, arg.split()))[0]
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must contain integers separated by space")

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--landmarkNum", type=int, default=19)
parser.add_argument("--image_scale", default=(800, 640), nargs='+', type=tuple_type)
parser.add_argument("--use_gpu", type=int, default=0)
parser.add_argument("--spacing", type=float, default=0.1)
parser.add_argument("--R1", type=int, default=41)
parser.add_argument("--R2", type=int, default=41)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--data_enhanceNum", type=int, default=1)
parser.add_argument("--stage", type=str, default="train")
parser.add_argument("--saveName", type=str, default="test1")
parser.add_argument("--testName", type=str, default="30cepha100_fusion_unsuper.pkl")
parser.add_argument("--dataRoot", type=str, default="process_data/")
parser.add_argument("--supervised_dataset_train", type=str, default="cepha/")
parser.add_argument("--supervised_dataset_test", type=str, default="cepha/")
parser.add_argument("--unsupervised_dataset", type=str, default="cepha/")
parser.add_argument("--trainingSetCsv", type=str, default="cepha_train.csv")
parser.add_argument("--testSetCsv", type=str, default="cepha_val.csv")
parser.add_argument("--unsupervisedCsv", type=str, default="cepha_val.csv")
parser.add_argument("--new_dataset", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--new_model", action=argparse.BooleanOptionalAction, default=False)



def main():
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.image_scale = tuple(config.image_scale)
    if config.new_model:
        model_ft = ViT(
            model_name='WinKawaks/vit-small-patch16-224',
            output_size=config.landmarkNum
        ).to(device)
    else:
        model_ft = models.fusionVGG19(
            torchvision.models.vgg19_bn(pretrained=True), config
        ).to(device)
    print("image scale ", config.image_scale)
    print("GPU: ", config.use_gpu)

    if not config.new_dataset:
        transform_origin=torchvision.transforms.Compose([
                        Rescale(config.image_scale),
                        ToTensor()
                        ])

        train_dataset_origin = LandmarksDataset(csv_file=config.dataRoot + config.trainingSetCsv,
                                                root_dir=config.dataRoot + config.supervised_dataset_train,
                                                transform=transform_origin,
                                                landmarksNum=config.landmarkNum
                                )

        val_dataset = LandmarksDataset(
                csv_file=config.dataRoot + config.testSetCsv,
                root_dir=config.dataRoot + config.supervised_dataset_test,
                transform=transform_origin,
                landmarksNum=config.landmarkNum
        )


        train_dataloader_t = DataLoader(train_dataset_origin, batch_size=config.batchSize,
                            shuffle=False, num_workers=18)

        val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                                shuffle=False, num_workers=18)
        train_dataloader = []
        val_dataloader = []
        # pre-load all data into memory for efficient training
        for data in tqdm(train_dataloader_t):
            train_dataloader.append(data)

        for data in tqdm(val_dataloader_t):
            val_dataloader.append(data)

    else:
        datamodule = LateralSkullRadiographDataModule(
            resized_image_size=config.image_scale,
            resized_point_reference_frame_size=config.image_scale,
            batch_size=config.batchSize,
            flip_augmentations=False,
        )

        train_dataloader = list(datamodule.train_dataloader())
        val_dataloader = list(datamodule.val_dataloader())

    print(len(train_dataloader), len(val_dataloader))

    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    para_list = list(model_ft.children())

    print("len", len(para_list))
    for idx in range(len(para_list)):
        print(idx, "-------------------->>>>", para_list[idx])

    # if use_gpu:
    model_ft = model_ft.to(device)
    criterion = lossFunction.HeatmapOffsetmapLoss(config)

    if config.new_model:
        optimizer_ft = optim.SGD(
            model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
        )
    else:
        optimizer_ft = optim.Adadelta(
            filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1.0
        )

    train.train_model(model_ft, dataloaders, criterion, optimizer_ft, config)


if __name__ == "__main__":
    main()
