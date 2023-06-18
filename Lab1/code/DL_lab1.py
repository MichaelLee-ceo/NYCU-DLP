import numpy as np
from utils import *
from model import SimpleNet

np.random.seed(0)

# x_train, y_train = generate_linear(100)
x_train, y_train = generate_XOR_easy()

dimensions = [2, 4, 4, 1]
model = SimpleNet(dim=dimensions, lr=0.03, activation="sigmoid", optimizer="sgd")
num_epoch = 10000

total_loss = []
for i in range(1, num_epoch+1):
    loss = 0
    for x, y in zip(x_train, y_train):
        output = model.forward_pass(x)
        model.backward_pass(x, output, y)
        loss += (mse(output, y).item())
    
    total_loss.append(loss / len(x_train))
    if i % 500 == 0 or i == 1:
        print('epoch', i, 'loss :', loss / len(x_train))

prediction = model.predict(x_train)
print('Accuracy:', np.equal(y_train.reshape(1, -1), prediction).sum() / len(y_train) * 100, '%')

show_train_loss(total_loss, num_epoch)
show_result(x_train, y_train, prediction)