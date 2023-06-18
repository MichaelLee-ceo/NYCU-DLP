import argparse
import torch
from BCIDataset import BCIDataset
from models.EEGNet import EEGNet
from models.DeepConvNet import DeepConvNet
from models.MyNet import MyNet
from utils import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='EEG Classification')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--activation', default='elu', type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using:', device)

trainset = BCIDataset(train=True)
testset = BCIDataset(train=False)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

model = MyNet().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

print('----- start training -----')
train_acc, test_acc = [], []
best_acc = 0.0
for epoch in range(args.num_epochs):
    total_loss = 0
    total, correct = 0, 0
    model.train()
    for idx, (data, label) in enumerate(trainLoader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_acc.append(100 * correct / total)

    # Evaluation on testset
    with torch.no_grad():
        model.eval()
        total_test, correct_test = 0, 0
        for idx, (data, label) in enumerate(testLoader):
            data, label = data.to(device), label.to(device)
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total_test += label.size(0)
            correct_test += (predicted == label).sum().item()
        test_acc.append(100 * correct_test / total_test)

    if test_acc[-1] > best_acc:
        best_acc = test_acc[-1]

    if epoch % 10 == 0 or epoch == args.num_epochs-1:
        print('Epoch: [{}/{}] loss: {:.5f}, acc: {:.2f}%, test_acc: {:.2f}%'. \
            format(epoch+1, args.num_epochs, total_loss / len(trainLoader), train_acc[-1], test_acc[-1]))
print('-- Best acc: {:.2f}%'.format(best_acc))