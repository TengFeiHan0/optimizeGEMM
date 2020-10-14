from  tvm import relay
import tvm.relay.op

x = relay.expr.var('x', relay.scalar_type('int64'), dtype= 'int64')
one = relay.expr.const(1, dtype= 'int64')
add = relay.op.tensor.add(x, one)
func = relay.Function([x], add, relay.scalar_type('int64'))

mod = tvm.ir.IRModule.from_expr(func)
graph, lib, params = tvm.relay.build(mod, 'llvm', params={})
print("TVM graph:\n", graph)
print("TVM parameters:\n", params)
print("TVM compiled target function:\n", lib.get_source())