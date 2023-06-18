import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataloader import read_bci_data

class BCIDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.transform = transform
        data = read_bci_data()
        if train:
            self.x = data[0]
            self.y = data[1]
        else:
            self.x = data[2]
            self.y = data[3]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        y = torch.Tensor([self.y[idx]]).long().squeeze(0)
        # y_onehot = F.one_hot(y, num_classes=2).squeeze(0)
        # print(y_onehot)
        return torch.Tensor(self.x[idx]), y