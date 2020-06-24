#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 5.reg-gluon.py 
@Time : 2020/6/24 16:44 
"""
# 高维线性回归数据集
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

# 定义训练和测试
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)  # ArrayDataset Dataset对象用于数据的收集、加载和变换
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size,
                                        shuffle=True)  # DataLoader对象加载Dataset，迭代时返回batch大小的样本,用于返回batch大小的数据。
square_loss = gluon.loss.L2Loss()  # 均方误差函数


def test(net, X, y):
    square_loss = gluon.loss.L2Loss()
    return square_loss(net(X), y).mean().asscalar()  # 将向量X转换成标量，且向量X只能为一维含单个元素的向量


def train(weight_decay):
    learning_rate = 0.005
    epochs = 10
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    # 注意到这里的‘wd’
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': weight_decay})
    # 权重：w = w-lr*grad - wd*w
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(test(net, X_train, y_train))
        test_loss.append(test(net, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    print('learned weight', net[0].weight.data(), 'learned bias', net[0].bias.data())
    plt.show()
    return


train(0)
