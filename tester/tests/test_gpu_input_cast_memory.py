"""输入 cast 的显存准入回归；仅创建配置，不分配大张量。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tester.input_generation.materialization import build_materialization_plan
from tester.input_generation.tensor_config import TensorConfig
from tester.runtime.gpu_memory_preflight import decide_gpu_memory_preflight, estimate_gpu_memory


@pytest.mark.parametrize("framework", ["paddle", "torch"])
@pytest.mark.parametrize("backend", ["numpy", "torch", "paddle"])
@pytest.mark.parametrize(
    "dtype,peak", [("bfloat16", 600), ("float8_e4m3fn", 300), ("float8_e5m2", 300)]
)
def test_cpu_source_cast_keeps_intermediate_on_gpu(framework, backend, dtype, peak):
    # CPU 源搬运后的中间量不属于 generated_inputs，必须由物化阶段持有。
    config = TensorConfig([100], dtype)
    plan = build_materialization_plan(config, backend, framework, input_source_on_gpu=False)
    assert plan.peak_bytes >= peak
    assert plan.persistent_bytes == config.nbytes()
    assert plan.source_bytes == 0


@pytest.mark.parametrize(
    "backend,framework", [("paddle", "paddle"), ("torch", "torch"), ("torch", "paddle")]
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn", "float8_e5m2"])
def test_gpu_source_is_counted_by_shared_stage_only(backend, framework, dtype):
    # 原生 GPU 源已在阶段公共组件中驻留；此处只新增 cast 目标。
    config = TensorConfig([100], dtype)
    plan = build_materialization_plan(config, backend, framework, input_source_on_gpu=True)
    assert plan.peak_bytes == config.nbytes()


@pytest.mark.parametrize(
    "dtype,shape", [("bfloat16", [25028608, 2048]), ("float8_e4m3fn", [50057216, 2048])]
)
def test_oversize_cast_is_rejected_without_generating_inputs(dtype, shape):
    # 最终输入小于整卡，但转换峰值大于整卡；不得进入耗时造数阶段。
    tensor = TensorConfig(shape, dtype)
    api = SimpleNamespace(api_name="paddle.reshape", args=[tensor], kwargs={})
    gpu = SimpleNamespace(enabled=True, memory_budget=268.59, comparison_memory_budget=0)
    assert tensor.nbytes() < int(gpu.memory_budget * 1024**3)
    decision = decide_gpu_memory_preflight(
        api, "paddle_only", gpu, check_grad=False, input_backend="numpy", input_source_on_gpu=False
    )
    assert decision.should_skip
    assert decision.rejected_stage.name == "framework_input_materialization"
    assert decision.rejected_stage.total_bytes > decision.capacity_bytes


@pytest.mark.parametrize("mode", ["accuracy_stable", "accuracy_stable_dual_gpu"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_stable_snapshot_rebuild_has_no_cast_intermediate(mode, dtype):
    # snapshot 保存最终 dtype，不能把首次物化的 cast 再套到重建阶段。
    config = TensorConfig([100], dtype)
    api = SimpleNamespace(api_name="paddle.reshape", args=[config], kwargs={})
    estimate = estimate_gpu_memory(
        api, mode, check_grad=False, input_backend="numpy", input_source_on_gpu=False
    )
    rebuilt = [
        stage
        for stage in estimate.stages
        if stage.name.endswith("_framework_input_materialization")
    ]
    assert len(rebuilt) == 2
    assert all(stage.total_bytes == config.nbytes() for stage in rebuilt)


def test_materialization_peak_retains_previous_inputs_and_excludes_cpu_place():
    # 顺序 cast 只持有当前临时量，但此前已完成的 GPU 输入仍然存活。
    first = TensorConfig([100], "bfloat16")
    second = TensorConfig([200], "bfloat16")
    cpu = TensorConfig([1000000], "bfloat16", place="cpu")
    api = SimpleNamespace(api_name="paddle.concat", args=[[first, second, cpu]], kwargs={})
    estimate = estimate_gpu_memory(
        api, "paddle_only", check_grad=False, input_backend="numpy", input_source_on_gpu=False
    )
    assert estimate.peak_stage.total_bytes == 200 + 1200
