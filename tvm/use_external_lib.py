import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing

out_channels = 16
batch_size = 1

data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
)
simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
simple_net = relay.nn.relu(simple_net)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)
net, params = testing.create_workload(simple_net)

import logging

logging.basicConfig(level=logging.DEBUG)
target = "cuda"

lib = relay.build_module.build(net, target, params=params)
ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()

out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cuda = out.asnumpy()



net, params = testing.create_workload(simple_net)
target = "cuda -libs=cudnn"  # use cudnn for convolution
lib = relay.build_module.build(net, target, params=params)
ctx = tvm.context(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cudnn = out.asnumpy()
tvm.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)