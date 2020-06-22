from mxnet import gluon

# 1.定义模型
net = gluon.nn.Sequential()  # 串行网络
with net.name_scope():
    net.add(gluon.nn.Flatten())  # 输入的图片转换成矩阵
    net.add(gluon.nn.Dense(256, activation='relu'))  # Dense Layer
    net.add(gluon.nn.Dense(10))
    print(net)
    net.initialize()

# 2.读取数据并训练
import sys

sys.path.append('..')
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd as autograd

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
