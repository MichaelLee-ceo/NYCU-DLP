from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from models.resnet import ResNet18, ResNet50
from dataloader import RetinopathyLoader
from utils import *

torch.manual_seed(0)
mkdir('figures')
mkdir('checkpoint')

parser = argparse.ArgumentParser(description="Diabetic Retinopathy Detection")
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--model', default="resnet18", type=str)
parser.add_argument('--pretrain', default=False, type=bool)
args = parser.parse_args()

device = getDevice()
if args.pretrain:
    if args.model == "resnet18":
        model = resnet18(weights="DEFAULT")
    else:
        model = resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 5)
else:
    if args.model == "resnet18":
        model = ResNet18()
    else:
        model = ResNet50()
model = model.to(device)

num_epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
}

trainset = RetinopathyLoader(root='./data/train/', mode="train", transforms=transforms['train'])
testset = RetinopathyLoader(root='./data/test/', mode="test", transforms=transforms['test'])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

train_total_loss, train_total_acc, test_total_loss, test_total_acc = [], [], [], []

best_acc = 0.0
state = {
    'model': [],
    'acc': 0,
    'epoch': 0,
}
print('\n----- start training -----')
for epoch in range(num_epochs):
    model.train()
    train_total = 0
    train_loss, train_correct = 0, 0
    for idx, (x, label) in enumerate(tqdm(train_loader)):
        x, label = x.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(x)

        predicted = torch.argmax(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_total_loss.append(train_loss / len(train_loader))
    train_total_acc.append(100 * train_correct / train_total)

    test_total = 0
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        model.eval()
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            predicted = torch.argmax(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

            test_loss += loss_fn(outputs, target).item()
    test_total_loss.append(test_loss / len(test_loader))
    test_total_acc.append(100 * test_correct / test_total)

    print('Epoch: {}/{}'.format(epoch+1, num_epochs))
    print('[Train] loss: {:.5f}, acc: {:.2f}%'.format(train_total_loss[-1], train_total_acc[-1]))
    print('[Test]  loss: {:.5f}, acc: {:.2f}%'.format(test_total_loss[-1], test_total_acc[-1]))

    # save checkpoint
    if test_total_acc[-1] > best_acc:
        best_acc = test_total_acc[-1]
        state['model'] = model.state_dict()
        state['acc'] = test_total_acc[-1]
        state['epoch'] = epoch
        print('- New checkpoint -')

torch.save(state, './checkpoint/' + args.model + "_" + str(args.pretrain))

print('\nBest acc: {:.2f}%'.format(state['acc']))
show_train_result(num_epochs, train_total_loss, train_total_acc, test_total_loss, test_total_acc, args.model + '_' + str(args.pretrain))

# evaluate on best model
# state = torch.load('./checkpoint/' + args.model + "_" + str(args.pretrain))
model.load_state_dict(state['model'])
y_true, y_pred = evaluate(model, test_loader, device)
plot_confusion_matrix(y_true, y_pred, 5, 'true', args.model + '_' + str(args.pretrain))