import torch

LABEL_DICT = {'bird': 0, 'car': 1, 'cat': 2, 'deer': 3, 'dog': 4, 'frog': 5, 'horse': 6, 'plane': 7, 'ship': 8, 'truck': 9}
LABEL_DICT_REVERSE = {0: 'bird', 1: 'car', 2: 'cat', 3: 'deer', 4: 'dog', 5: 'frog', 6: 'horse', 7: 'plane', 8: 'ship', 9: 'truck'}

DATASET_PATH = 'cifar10'
SAVE_MODEL_PATH = 'weights'
RESIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10

