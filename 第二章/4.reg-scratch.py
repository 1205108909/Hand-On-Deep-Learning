# 正则化

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 20
num_test = 100
num_inputs = 200

# 生成数据集-模型真实参数
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

# 生产训练和测试数据集
X = nd.random.normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w)
y += 0.01 * nd.random.normal(shape=y.shape)
X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

import random

batch_size = 1


def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)  # shuffle() 方法将序列的所有元素随机排序
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield X.take(j), y.take(j)


# 初始化模型参数
def get_params():
    w = nd.random.normal(shape=(num_inputs, 1)) * 0.1
    b = nd.zeros((1,))
    for param in (w, b):
        param.attach_grad()
    return (w, b)


# L2范数正则化
def net(X, lambd, w, b):
    return nd.dot(X, w) + b + lambd * ((w ** 2).sum() + b ** 2)


# 3.定义训练和测试步骤
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):  # lr 即 learning_rate
    for param in params:
        param[:] = param - lr * param.grad


def test(params, X, y):
    return square_loss(net(X, 0, *params), y).mean().asscalar()


def train(lambd):
    epochs = 10
    learning_rate = 0.002
    params = get_params()  # 权重信息
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, lambd, *params)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
        train_loss.append(test(params, X_train, y_train))
        test_loss.append(test(params, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])

    print('learned w[:10]:', params[0][:10], 'learned b:', params[1])
    plt.show()
    return

#使用正则化
train(2)

