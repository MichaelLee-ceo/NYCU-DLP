import torch
import torch.nn as nn

torch.manual_seed(0)

class MyNet(nn.Module):
    def __init__(self, num_channel=6):
        super(MyNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, num_channel, kernel_size=(1, 5), padding=2),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, kernel_size=(1, 5), padding=0),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(num_channel, num_channel*2, kernel_size=(1, 5), padding=0),
            nn.BatchNorm2d(num_channel*2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5),
        )
    
        self.fc = nn.Sequential(
            nn.Linear(num_channel * 40 * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
