import torch
import torch.nn as nn

torch.manual_seed(0)

class DeepConvNet(nn.Module):
    def __init__(self, activation="elu"):
        super(DeepConvNet, self).__init__()

        if activation == "elu":
            activation_fn = nn.ELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "leakyRelu":
            activation_fn = nn.LeakyReLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25),
            activation_fn,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.block2 = self.make_block(in_channel=25, out_channel=50, activation=activation_fn)
        self.block3 = self.make_block(in_channel=50, out_channel=100, activation=activation_fn)
        self.block4 = self.make_block(in_channel=100, out_channel=200, activation=activation_fn)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 43, 2),
        )

    def make_block(self, in_channel, out_channel, activation):
        block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 5)),
            nn.BatchNorm2d(out_channel),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        return block
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)

        return x