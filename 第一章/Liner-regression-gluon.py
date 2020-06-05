#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : Liner-regression-gluon.py
@Time : 2020/6/5 7:39
"""

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

# 1.创建数据集合
X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

# 2.数据读取
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(
    dataset, batch_size, shuffle=True)  # 从数据集中加载数据并返回小批量数据

for data, label in data_iter:
    print(data)
    print(label)
    break

# 3.定义模型
net = gluon.nn.Sequential()  # Sequential容器
net.add(gluon.nn.Dense(1))  # 在Sequential容器上加入一层
print(net)

# 4.初始化模型参数
net.initialize()

# 5.损失函数
square_loss = gluon.loss.L2Loss()

# 6.优化
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

#7.训练
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)

        total_loss = nd.sum(loss).asscalar()
    print("Epoch %d,averge loss:%f" % (e, total_loss / num_examples))

#8.取结果
dense = net[0]
print(dense.weight.data())
print(dense.bias.data())

help(dense.weight)
