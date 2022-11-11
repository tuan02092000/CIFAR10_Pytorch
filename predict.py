import torch

import config
from lib import *
from dataset import CIFAR10
from my_utils import make_data, split_data, visualize_model
from dataloader import get_dataloader_dict
from model import CIFARModel
from transform import ImageTransform

if __name__ == '__main__':
    # Dataset
    data = make_data(config.DATASET_PATH)
    datasets = split_data(data)
    train_data = CIFAR10(datasets[0], transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='train')
    val_data = CIFAR10(datasets[1], transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='val')
    test_data = CIFAR10(datasets[2], transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='test')

    # Dataloader
    dataloader = get_dataloader_dict(train_data, val_data, test_data, batch_size=config.BATCH_SIZE)

    # Model
    cifar_model = CIFARModel()
    cifar_model.to(config.DEVICE)
    cifar_model = torch.load('weights/cifar10.pt')

    visualize_model(cifar_model, dataloader['test'])



