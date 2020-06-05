#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : Softmax_regression_scratch.py
@Time : 2020/6/5 8:21
"""

import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import ndarray as nd


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

data, label = mnist_train[0]


def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def get_text_labels(label):
    text_labels = [
        't-shirt',
        'trouser',
        'pullover',
        'dress',
        'coat',
        'sandal',
        'shirt',
        'sneaker',
        'bag',
        'ankle boot']
    return [text_labels[int(i)] for i in label]


data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))

# 1.数据读取
batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train, batch_size, shuffle=True)  # 用来训练模型
test_data = gluon.data.DataLoader(
    mnist_test, batch_size, shuffle=False)  # 看模型效果如何

# 2.初始化模型参数
num_inputs = 28 * 28  # 一张图片是28*28*1
num_outputs = 10  # 输出10张

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [W, b]

for param in params:
    param.attach_grad()  # 对模型参数附上梯度

# 3.定义模型
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition

x= nd.random_normal(shape=(2,5))
x_prob = softmax(x)
print(f'x={x}')
print(f'x_prob ={x_prob}')
print(f'x_probsum = {x_prob.sum(axis=1)}')

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b) #-1系统自己决定行数
