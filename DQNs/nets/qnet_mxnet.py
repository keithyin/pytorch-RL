import mxnet as mx
import itertools


def demo_bind():
    x = mx.sym.Variable("x")
    y = x + 1
    x_array = mx.nd.array([1., 2.], ctx=mx.cpu())
    executor = y.bind(ctx=mx.cpu(), args=[x_array])
    executor.forward()
    print(executor.outputs[0].asnumpy())
    x_array[1] = 3.
    executor.forward(is_train=True)
    print(executor.outputs[0].asnumpy())
    executor.backward(out_grads=[mx.nd.array([1., 1.])])
    print(executor.grad_arrays)


def demo_symbol():
    inputs = mx.sym.Variable(name='inputs', shape=(32, 4, 84, 84))
    net = mx.sym.Convolution(data=inputs, kernel=(8, 8), stride=(4, 4), pad=(2, 2), num_filter=32)
    net = mx.sym.flatten(data=net)
    executor = net.simple_bind(ctx=mx.cpu())
    executor.forward()
    print(executor.outputs[0].asnumpy().shape)


class QNetwork(object):
    def __init__(self, num_classes, batch_size=32):
        self.func = None
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.build_func()

    def build_func(self):
        inputs = mx.sym.Variable(name='inputs', shape=(self.batch_size, 4, 84, 84))
        net = mx.sym.Convolution(data=inputs, kernel=(8, 8), stride=(4, 4), pad=(2, 2), num_filter=32)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.Convolution(data=net, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=64)
        net = mx.sym.flatten(data=net)
        net = mx.sym.FullyConnected(data=net, num_hidden=512)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.FullyConnected(data=net, num_hidden=self.num_classes)
        self.func = net.simple_bind(ctx=mx.gpu())

    def forward(self, is_train=False, **kwargs):
        self.func.forward(is_train=is_train, **kwargs)
        return self.func.outputs

    def backward(self, out_grads=None):
        self.func.backward(out_grads=out_grads)
        pass


def main():
    # Q_net = QNetwork(num_classes=6, batch_size=32)
    out_grads = mx.nd.ones(shape=(32, 6), ctx=mx.gpu())
    a = mx.nd.zeros(shape=(32, 6), ctx=mx.cpu())
    a.copyto(out_grads)
    print(out_grads.asnumpy())
    # for i in itertools.count():
    #     print("step %d" % i)
    #     Q_net.forward(is_train=True)
    #     Q_net.backward(out_grads=[out_grads])


if __name__ == '__main__':
    print(mx.cpu())
    main()
