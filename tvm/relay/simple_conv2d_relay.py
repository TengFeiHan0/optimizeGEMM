import tvm
from  tvm import relay
import numpy as np
from tvm.contrib import graph_runtime
from tvm.relay import transform

def batch_norm_infer(data,
                     gamma =None,
                     beta =None,
                     moving_mean =None,
                     moving_var =None,
                     **kwargs
                     ):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                               gamma=gamma,
                               beta=beta,
                               moving_mean=moving_mean,
                               moving_var=moving_var,
                               **kwargs)[0]

def conv2d(data, weight=None, **kwargs):
    name = kwargs.get('name')
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)    
    
def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), eps=1e-5):
    conv = conv2d(data=data, 
                  channels=channels,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  data_layout='NCHW',
                  name = name+'_conv'
                )
    bn = batch_norm_infer(data=conv, epsilon=eps, name=name+'_bn')
    act = relay.nn.relu(data=bn)
    return act


data_shape = (1, 3, 224, 224)
#kernel_shape = (32, 3, 3, 3)
dtype = "float32"
data = relay.var("data", shape=data_shape, dtype=dtype)
act = conv_block(data, "graph", 32, strides=(2, 2))
func = relay.Function(relay.analysis.free_vars(act),act)


mod = tvm.ir.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)
shape_dict = {
    v.name_hint : v.checked_type for v in mod["main"].params}
np.random.seed(0)
params = {}
for k, v in shape_dict.items():
    if k == "data":
        continue
    init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
    params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))

target = "llvm"
ctx = tvm.context(target, 0)

#print("Relay module function:\n", mod.astext(show_meta_data=False))
#print("TVM parameters:\n", params.keys())

# with relay.build_config(opt_level=3):
#   import pdb
#   graph, lib, params = relay.build(mod, target, params=params)

# print("TVM graph:\n", graph)
# print("TVM parameters:\n", params.keys())
# # print("TVM compiled target function:\n", lib.get_source())
# module = graph_runtime.create(graph, lib, ctx)
# data_tvm = tvm.nd.array((np.random.uniform(-1, 1, size=data_shape)).astype(dtype))
# module.set_input('data', data_tvm)
# module.set_input(**params)
# module.run()
# output = module.get_output(0)

def _bind_params(func, params):
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.expr.const(v)
    return relay.expr.bind(func, bind_dict)

def relay_optimize(func, params=None):
    if params:
        graph = _bind_params(func, params)
    
    optimize = tvm.transform.Sequential(
        [relay.transform.SimplifyInference(),
                                      relay.transform.FoldConstant(),
                                      relay.transform.FoldScaleAxis(),
                                      relay.transform.CanonicalizeOps(),
                                      relay.transform.FoldConstant(),
                                      ]
    )
    mod = tvm.ir.IRModule.from_expr(graph)
    mod = optimize(mod)
    return mod["main"]

mod['main'] = relay_optimize(mod['main'], params)
print("Relay module function:\n", mod.astext(show_meta_data=False))
    


