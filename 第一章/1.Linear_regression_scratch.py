#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 1.Linear_regression_scratch.py
@Time : 2020/6/5 8:59
"""

import random
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

# 1.创建数据集
num_inputs = 2
num_example = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_example, num_inputs))
Y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
Y += 0.01 + nd.random_normal(shape=Y.shape)

# 2.数据读取
batch_size = 10

def data_iter():
    idx = list(range(num_example))
    random.shuffle(idx)
    for i in range(0, num_example, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_example)])
        yield nd.take(X, j), nd.take(Y, j)

for data, label in data_iter():
    print(data, label)
    break

# 3.初始化模型
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros(1)
params = [w, b]

for param in params:
    param.attach_grad()

# 4.定义模型
def net(X):
    return nd.dot(X, w) + b
print(net(data))


# 5.损失函数
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape))**2

# 6.优化
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 7.训练
epochs = 10
learning_rate = 0.001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)

        total_loss = nd.sum(loss).asscalar()
    print("Epoch %d,averge loss:%f" % (e, total_loss / num_example))
