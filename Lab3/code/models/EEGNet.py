import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, channel=16, activation='elu'):
        super(EEGNet, self).__init__()
        if activation == 'elu':
            activation_fn = nn.ELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'leakyRelu':
            activation_fn = nn.LeakyReLU()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(channel),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(channel*2),
            activation_fn,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25),
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(channel*2),
            activation_fn,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25),
        )

        self.classify = nn.Sequential(
            nn.Linear(736, 2),
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)

        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)

        return x