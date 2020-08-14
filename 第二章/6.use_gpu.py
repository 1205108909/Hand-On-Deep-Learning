from mxnet import ndarray as nd
import mxnet as mx

x = nd.array([1, 2, 3], ctx=mx.cpu())
print(x.context)
print('-----')

y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())
print((y, z))

print('-----')

yy = y.as_in_context(mx.gpu())
zz = z.copyto(mx.gpu())
print((yy is y, zz is z))
