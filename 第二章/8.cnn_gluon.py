#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 8.cnn_gluon.py 
@Time : 2020/7/13 10:56 
"""
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation="relu"))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation='relu'))
    net.add(gluon.nn.Dense(10))

import sys

sys.path.append('..')
import utils

ctx = utils.try_gpu()
net.initialize(ctx=ctx)

from mxnet import autograd as autograd
from utils import SGD, accuracy, exaluate_accuracy
from mxnet import gluon
from mxnet import ndarray as nd

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        
        train_acc += accuracy(output, label)
        train_loss += nd.mean(loss).asscalar()

    test_acc = exaluate_accuracy(test_data, net, ctx)
    print('Epoch %d. Loss:%f,Train ass %f,Test acc %f' % (
        epoch, train_loss / len(train_acc), train_acc / len(train_data), test_acc / len(test_data)))
