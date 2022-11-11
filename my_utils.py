import os

import matplotlib.pyplot as plt
import torch

import config
from lib import *

def make_data(dataset_path):
    datasets = {'path': [], 'label': []}
    for label in os.listdir(dataset_path):
        for image in os.listdir(os.path.join(dataset_path, label)):
            datasets['path'].append(os.path.join(dataset_path, label, image))
            datasets['label'].append(label)
    return datasets

def split_data(datasets):
    train_image, val_test_image, train_label, val_test_label = train_test_split(datasets['path'], datasets['label'], test_size=0.3, random_state=42)
    val_image, test_image, val_label, test_label = train_test_split(val_test_image, val_test_label, test_size=0.3, random_state=42)
    train_data = {'path': train_image, 'label': train_label}
    val_data = {'path': val_image, 'label': val_label}
    test_data = {'path': test_image, 'label': test_label}
    return train_data, val_data, test_data

def save_model(model, name_model):
    torch.save(model, os.path.join(config.SAVE_MODEL_PATH, name_model))

def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            images_so_far = 0
            for j in range(inputs.size()[0]):
                images_so_far += 1
                if images_so_far == 7:
                    break
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {config.LABEL_DICT_REVERSE[preds[j].item()]}')
                # print(config.LABEL_DICT_REVERSE[preds[j].item()])
                plt.imshow(inputs.cpu().data[j].permute(1, 2, 0))
            plt.show()
        model.train(mode=was_training)

if __name__ == '__main__':
    data = make_data(config.DATASET_PATH)
    train, val, test = split_data(data)
    print('[INFO] Len of train: ', len(train['path']))
    print('[INFO] Len of val: ', len(val['path']))
    print('[INFO] Len of test: ', len(test['path']))
