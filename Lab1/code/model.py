import numpy as np
from utils import *

np.random.seed(0)

class Dense():
    def __init__(self, in_dim, out_dim, activation="sigmoid"):
        self.layer = self.linear(in_dim, out_dim)
        self.grad = [np.zeros(self.layer[0].shape), np.zeros(self.layer[1].shape)]
        self.m = [np.zeros(self.layer[0].shape), np.zeros(self.layer[1].shape)]         # for momentum
        self.v = [np.zeros(self.layer[0].shape), np.zeros(self.layer[1].shape)]         # for RMSprop
        self.t = 0

        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "relu":
            self.activation = relu
        elif activation == "tanh":
            self.activation = tanh
        else:
            self.activation = identity

        self.Z = np.zeros((out_dim, 1))
        self.A = np.zeros((out_dim, 1))
        self.dc_dz = np.zeros((out_dim, 1))

    def linear(self, input_dim, output_dim):
        w = np.random.normal(0, 1, (output_dim, input_dim))
        b = np.random.normal(0, 1, (output_dim, 1))
        return [w, b]

    def update(self, new_grad):
        self.layer[0] -= new_grad[0]
        self.layer[1] -= new_grad[1]


class SimpleNet():
    def __init__(self, dim=[2, 1], lr=0.001, activation="sigmoid", optimizer="sgd"):
        self.layers = self.build_layer(dim, activation)
        self.num_layers = len(self.layers)
        self.lr = lr
        self.optimizer = optimizer

    # 建每一層 network 中間的 weight 跟 bias (training 要 optimize 的對象)
    def build_layer(self, dim, activation):
        io_dims = list(zip(dim[:-1], dim[1:]))        # [(32, 64), (64, 128)]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(io_dims):
            layers.append(Dense(in_dim, out_dim, activation))
        return layers
    
    # forward pass (把 activation 的 output 算出來，之後 backpropagation 要用到)
    def forward_pass(self, x):
        a = x.reshape(-1, 1)
        for i in range(self.num_layers):
            w, b = self.layers[i].layer
            z = np.dot(w, a) + b
            a = self.layers[i].activation(z)
            self.layers[i].Z = z
            self.layers[i].A = a
        return a
    
    # backward pass (把每一層的 dC/dz 算出來，然後算 gradient)
    def backward_pass(self, x, y, y_hat):
        # 計算 dC/dz
        self.layers[-1].dc_dz = mse(y, y_hat, derivative=True) * self.layers[-1].activation(self.layers[-1].A, derivative=True)
        for i in range(self.num_layers-2, -1, -1):
            self.layers[i].dc_dz = self.layers[i].activation(self.layers[i].A, derivative=True) * np.dot(self.layers[i+1].layer[0].T, self.layers[i+1].dc_dz)

        # 計算 gradient = a (前一層的 activation output) * dc_dz (當前這一層的微分)
        for i in range(self.num_layers):
            self.layers[i].grad[0] = x.T * self.layers[i].dc_dz
            self.layers[i].grad[1] = self.layers[i].dc_dz
            x = self.layers[i].A

        # 用 optimizer 去優化 gradient descent
        self.step()

    def step(self):
        for i in range(self.num_layers):
            new_grad = [0, 0]
            for j in range(2):                      # iterate for [weight, bias]
                if self.optimizer == "momentum":
                    self.layers[i].m[j] = (0.9 * self.layers[i].m[j]) - (self.lr * self.layers[i].grad[j])
                    new_grad[j] = -self.layers[i].m[j]
                elif self.optimizer == "adam":
                    self.layers[i].t += 1
                    self.layers[i].m[j] = (0.9 * self.layers[i].m[j]) + (1 - 0.9) * self.layers[i].grad[j]
                    self.layers[i].v[j] = (0.999 * self.layers[i].v[j]) + (1 - 0.999) * self.layers[i].grad[j]**2
                    m, v = [0, 0], [0, 0]
                    m[j] = self.layers[i].m[j] / (1 - (0.9)**self.layers[i].t)
                    v[j] = self.layers[i].v[j] / (1 - (0.999)**self.layers[i].t)
                    new_grad[j] = self.lr * m[j] / (np.sqrt(v[j]) + 1e-8)
                elif self.optimizer == "sgd":
                    new_grad[j] = self.lr * self.layers[i].grad[j] 
            self.layers[i].update(new_grad)

    def predict(self, X):
        prediction = []
        for x in X:
            output = self.forward_pass(x)
            if output > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction