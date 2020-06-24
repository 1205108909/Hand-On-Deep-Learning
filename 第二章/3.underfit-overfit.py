from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

x = nd.random.normal(shape=(num_train + num_test, 1))
X = nd.concat(x, nd.power(x, 2), nd.power(x, 3))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
y += 0.1 * nd.random.normal(shape=y.shape)
y_train, y_test = y[:num_train], y[num_train:]
print(x[:5])
print('----------------')
print(X[:5])
print('----------------')
print(y[:5])

# 3.定义训练和测试步骤
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def test(net, X, y):
    square_loss = gluon.loss.L2Loss()
    return square_loss(net(X), y).mean().asscalar()  # 将向量X转换成标量，且向量X只能为一维含单个元素的向量


def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()

    # 设置默认参数
    learning_rate = 0.01
    epochs = 100
    batch_size = 1

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)  # ArrayDataset Dataset对象用于数据的收集、加载和变换
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size,
                                            shuffle=True)  # DataLoader对象加载Dataset，迭代时返回batch大小的样本,用于返回batch大小的数据。
    # 默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    square_loss = gluon.loss.L2Loss()  # 均方误差函数

    # 保存训练和测试损失
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter_train:
            # 默认条件下，MXNet不会自动记录和‘构建’用于求导的计算图，我们需要使用autograd里的record()函数来显式的要求MXNet记录我们需要求导的程序
            # 使用x.attach_grad()为梯度分配空间，调用with autograd.record()计算梯度，再使用backward()进行反传
            with autograd.record():  # 需要反向传导的地方记得 record 一下
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()  # 计算梯度
            trainer.step(batch_size)  # 更新模型参数
        train_loss.append(square_loss(net(X_train), y_train).mean().asscalar())
        test_loss.append(square_loss(net(X_test), y_test).mean().asscalar())

    # 打印结果
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    print('learned weight', net[0].weight.data(), 'learned bias', net[0].bias.data())
    plt.show()
    return

#三项多项式正常拟合
train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])
#欠拟合 只用一层
train(x[:num_train, :], x[num_train:, :], y[:num_train], y[num_train:])
#过拟合
train(X[:2, :], X[num_train:, :], y[:2], y[num_train:]) #只给两个训练样本
