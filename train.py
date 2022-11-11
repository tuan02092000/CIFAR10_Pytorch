import copy
from lib import *
import config
from transform import ImageTransform
from model import CIFARModel
from dataset import CIFAR10
from dataloader import get_dataloader_dict
from my_utils import make_data, split_data, save_model

def train_model(model, criterion, optimier, dataloader_dict, train_dataset, val_dataset):
    print('[INFO] start training network...')
    start = time.time()
    model = model.to(config.DEVICE)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        print('\n[INFO] Epoch {}/{}'.format(epoch, config.EPOCHS - 1))
        print('-' * 50)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels in tqdm(dataloader_dict[phase]):
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                optimier.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicts = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimier.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(predicts == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)
            else:
                epoch_loss = running_loss / len(val_dataset)
                epoch_acc = running_corrects.double() / len(val_dataset)

            print('\n[INFO] {} Loss: {:4f}, Acc: {:4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print(f'\n[INFO] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\n[INFO] Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    save_model(model, 'cifar10.pt')

if __name__ == '__main__':
    # datasets
    datasets = make_data(config.DATASET_PATH)
    train, val, test = split_data(datasets)
    train_data = CIFAR10(train, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='train')
    val_data = CIFAR10(val, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='val')
    test_data = CIFAR10(test, transform=ImageTransform(config.RESIZE, config.MEAN, config.STD), phase='test')

    # Dataloader
    dataloader_dict = get_dataloader_dict(train_data, val_data, test_data, batch_size=config.BATCH_SIZE)

    # Model
    cifar_model = CIFARModel()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(params=cifar_model.parameters(), lr=config.LR)

    # Train
    train_model(cifar_model, criterion, optimizer, dataloader_dict, train_data, val_data)
