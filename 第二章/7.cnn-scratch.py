# 卷积神经网络
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

# # pooling 池化层
# data = nd.arange(18).reshape((1, 2, 3, 3))
# max_pool = nd.pooling(data=data, pool_type='max', kernel=(2, 2))
# max_pool = nd.pooling(data=data, pool_type='max', kernel=(2, 2))

# 获取数据
import sys

sys.path.append('..')
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 定义模型
import mxnet as mx

try:
    ctx = mx.gpu()
    nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx

weight_scale = 0.01
num_output = 10

# output channels = 20,kernel = (5,5)
W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

# output channels = 50 ,kernel = (3,3)
W2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(W2.shape[0], ctx=ctx)

# output dim = 128
W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

# output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
    param.attach_grad()


# 卷积模块通常是“卷积层-激活层-池化层”。然后转成2D矩阵输出后给后面的全连接层。
def net(X, verbose=False):
    X = X.as_in_context(W1.context)
    # 第一层卷积
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))

    # 第二层卷积
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
    h2 = nd.flatten(h2)

    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)

    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block', h1.shape)
        print('2nd conv block', h2.shape)
        print('1st conv block', h3.shape)
        print('2nd conv block', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear


for data, _ in train_data:
    net(data, verbose=True)
    break

#     训练

from mxnet import autograd as autograd
from utils import SGD, accuracy, exaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = 0.2

for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate / batch_size)

        train_acc += accuracy(output, label)
        train_loss += nd.mean(loss).asscalar()

    test_acc = exaluate_accuracy(test_data, net, ctx)
    print('Epoch %d. Loss:%f,Train ass %f,Test acc %f' % (
    epoch, train_loss / len(train_acc), train_acc / len(train_data), test_acc / len(test_data)))
