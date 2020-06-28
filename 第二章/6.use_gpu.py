from mxnet import ndarray as nd
import mxnet as mx

x = nd.array([1, 2, 3], ctx=mx.gpu())
print(x.context)
