import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


target = tvm.target.Target("llvm -mcpu=core-avx2")
task = tvm.auto_scheduler.create_task(matmul_add, (128, 128, 128, "float32"), target)

# Inspect the computational graph
print(task.compute_dag)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000, measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")]
)

sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)
print(tvm.lower(sch, args, simple_mode=True))

func = tvm.build(sch, args)
a_np = np.random.uniform(size=(128, 128)).astype(np.float32)
b_np = np.random.uniform(size=(128, 128)).astype(np.float32)
c_np = np.random.uniform(size=(128, 128)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

ctx = tvm.cpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)
c_tvm = tvm.nd.array(c_np, ctx=ctx)
out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)

# Load the measuremnt record for the best schedule
inp, res = auto_scheduler.load_best("matmul.json", task.workload_key)

# Print equivalent python schedule API. This can be used for debugging and
# learning the behavior of the auto-scheduler.
print("Equivalent python schedule:")
print(task.compute_dag.print_python_code_from_state(inp.state))

# Rebuild the binary. This shows how you can apply the best schedule from a
# log file without reruning the search again.
sch, args = task.compute_dag.apply_steps_from_state(inp.state)
func = tvm.build(sch, args)

def resume_search(task, log_file):
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)


resume_search(task, "matmul.json")
    
