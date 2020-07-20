#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : 1.Block.py 
@Time : 2020/7/15 7:44 
"""

from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
print(net)


# 使用nn.Block来定义
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)  # 输出为10

    # 定义网络的计算 Dense0->relu->Dense1
    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))


net2 = MLP()
print(net2)
net2.initialize()
x = nd.random_uniform(shape=(4, 20))  # 输入为4
y = net2(x)
print(y)  # 输入为4，输出为10
