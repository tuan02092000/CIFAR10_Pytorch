import torch.utils.data
import config
from lib import *

def get_dataloader_dict(train_data, val_data, test_data, batch_size):
    return {'train': torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 num_workers=os.cpu_count(),
                                                 pin_memory=True if config.DEVICE == "cuda" else False,
                                                 shuffle=True,
                                                 drop_last=True),
            'val': torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size,
                                               num_workers=os.cpu_count(),
                                               pin_memory=True if config.DEVICE == "cuda" else False,
                                               shuffle=False,
                                               drop_last=True),
            'test': torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True if config.DEVICE == "cuda" else False,
                                                shuffle=False,
                                                drop_last=True)
            }
