#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 1.mlp-scratch.py 
@Time : 2020/6/22 16:29 
"""

import sys

sys.path.append('..')
from mxnet import ndarray as nd
from mxnet import gluon

num_inputs = 28 * 28
num_outputs = 10

num_hidden = 256
weight_scale = 0.1

w1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

w2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()


# 1.激活函数Relu (非线性)
def relu(X):
    return nd.maximum(X, 0)


# 2.定义模型 X是一张图片
def net(X):
    X = X.reshape((-1, num_inputs))  # 输入层
    h1 = relu(nd.dot(X, w1) + b1)  # 第一层 与 非线性点层 进入下一层
    output = nd.dot(h1, w2) + b2
    return output


# 3.Softmax和交叉熵损失函数
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
