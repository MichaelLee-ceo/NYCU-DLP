import numpy as np
import matplotlib.pyplot as plt

def show_result(num_epoch, train_acc, test_acc, model_name):
    x = np.arange(0, num_epoch)
    plt.figure(figsize=(10, 5))
    plt.title('Activation function comparision (' + model_name +')')
    plt.plot(x, train_acc[0], label='relu_train')
    plt.plot(x, test_acc[0], label='relu_test')

    plt.plot(x, train_acc[1], label='leaky_relu_train')
    plt.plot(x, test_acc[1], label='leaky_relu_test')

    plt.plot(x, train_acc[2], label='elu_train')
    plt.plot(x, test_acc[2], label='elu_test')
    plt.legend()
    plt.savefig(model_name + '.png')
    # plt.show()
