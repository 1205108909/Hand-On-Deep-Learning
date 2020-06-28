# 卷积神经网络
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

# pooling 池化层
data = nd.arange(18).reshape((1, 2, 3, 3))
max_pool = nd.pooling(data=data, pool_type='max', kernel=(2, 2))
max_pool = nd.pooling(data=data, pool_type='max', kernel=(2, 2))
