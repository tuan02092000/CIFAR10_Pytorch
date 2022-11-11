from lib import *
import config
from my_utils import make_data, split_data
from transform import ImageTransform

class CIFAR10:
    def __init__(self, datasets, transform=None, phase='train'):
        self.datasets = datasets
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.datasets['path'])
    def __getitem__(self, idx):
        image_path = self.datasets['path'][idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image, self.phase)
        label = torch.tensor(config.LABEL_DICT[self.datasets['label'][idx]])
        return image, label

if __name__ == '__main__':
    datasets = make_data(config.DATASET_PATH)
    train, val, test = split_data(datasets)
    train_data = CIFAR10(train, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='train')
    val_data = CIFAR10(val, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='val')
    test_data = CIFAR10(test, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='test')
    print('[INFO] Len of train: ', len(train_data))
    print('[INFO] Len of val: ', len(val_data))
    print('[INFO] Len of test: ',len(test_data))

