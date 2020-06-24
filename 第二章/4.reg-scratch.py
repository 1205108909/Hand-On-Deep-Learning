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
    random.shuffle(idx) #shuffle() 方法将序列的所有元素随机排序
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield X.take(j), y.take(j)



