#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 4.dropout-scratch.py 
@Time : 2020/7/21 7:45 
"""

# 丢弃法-过拟合
from mxnet.gluon import nn

help(nn.Dense)

# 实现dropout
from mxnet import nd


def dropout(X, dropout_probability):  # dropout_probability:dropout的概率
    keep_probability = 1 - dropout_probability
    assert 0 <= keep_probability <= 1

    if keep_probability == 0:
        return X.zeros_like()

    mask = nd.random.uniform(0, 1, X.shape, ctx=X.context) < keep_probability
    # 保证E[dropout(X)] == X
    scale = 1 / keep_probability
    return mask * X * scale


A = nd.arange(20).reshape((5, 4))
print(dropout(A, 0.5))

# 通常dropout在全连接后面
drop_prob1 = 0.2
drop_prob2 = 0.5


def net(X):
    X = X.reshape((-1, num_inputs))
    # 第一层全连接
    h1 = nd.relu(nd.dot(X, W1) + b1)
    h1 = dropout(h1, drop_prob1)
    h2 = nd.relu(nd.dot(X, W2) + b2)
    h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, W3) + b3

#85
