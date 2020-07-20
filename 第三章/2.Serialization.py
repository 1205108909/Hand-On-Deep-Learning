#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 2.Serialization.py 
@Time : 2020/7/20 8:12 
"""

from mxnet import nd
from mxnet.gluon import nn


def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(2))
    return net


net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2, 20))
print(net(x))
