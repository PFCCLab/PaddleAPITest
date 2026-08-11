"""engineV4 输入运行时治理的 CPU-only 回归测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy


def _numpy_states_equal(left, right):
    return left[0] == right[0] and numpy.array_equal(left[1], right[1]) and left[2:] == right[2:]


class RuntimeGovernanceTest(unittest.TestCase):
    def test_prepare_input_backend_does_not_consume_global_rng(self):
        # 只导入输入生成层，测试不触发 engine worker 或 GPU 初始化。
        from tester.input_generation import backend as backend_module

        prepare_input_backend = getattr(backend_module, "prepare_input_backend", None)
        self.assertIsNotNone(prepare_input_backend)
        resolve_input_backend_policy = backend_module.resolve_input_backend_policy

        policy = resolve_input_backend_policy(
            requested="numpy",
            use_gpu_mode=False,
            mode="paddle_only",
        )
        numpy.random.seed(2026)
        # preparation 的常量探针不应推进配置级 NumPy 状态。
        before = numpy.random.get_state()
        backend = prepare_input_backend(policy)
        after = numpy.random.get_state()

        # preparation 只能建立已有 backend 的通道，不能改变配置随机流。
        self.assertEqual(backend.name, "numpy")
        self.assertTrue(_numpy_states_equal(before, after))

    def test_output_grad_stream_is_reproducible_and_global_rng_free(self):
        # output grad 与 forward input 共用 backend 协议，但必须使用独立 stream。
        from tester.input_generation import backend as backend_module

        generate_output_grad = getattr(backend_module, "generate_output_grad", None)
        self.assertIsNotNone(generate_output_grad)

        numpy.random.seed(2026)
        before = numpy.random.get_state()
        first = generate_output_grad(
            dtype="float32",
            shape=(8,),
            backend_name="numpy",
            device="cpu",
            seed=9,
            config_fingerprint="case",
            stream_index=0,
        )
        after = numpy.random.get_state()
        # 相同 seed、config 和 stream 必须产生稳定值，便于跨 worker 对齐。
        second = generate_output_grad(
            dtype="float32",
            shape=(8,),
            backend_name="numpy",
            device="cpu",
            seed=9,
            config_fingerprint="case",
            stream_index=0,
        )
        # 只改变 stream 序号即可得到另一组值，避免多输出梯度相同。
        other_stream = generate_output_grad(
            dtype="float32",
            shape=(8,),
            backend_name="numpy",
            device="cpu",
            seed=9,
            config_fingerprint="case",
            stream_index=1,
        )

        self.assertTrue(_numpy_states_equal(before, after))
        numpy.testing.assert_array_equal(first, second)
        self.assertFalse(numpy.array_equal(first, other_stream))

    def test_gpu_slots_are_constructed_breadth_first(self):
        # WorkerPool 构造本身是 CPU-only，适合验证公平启动顺序。
        from engineV4 import WorkerPool

        options = SimpleNamespace(accuracy_dual_gpu=False, accuracy_stable_dual_gpu=False)
        pool = WorkerPool(
            [0, 1],
            {0: 2, 1: 2},
            options,
            gpu_total_memory_map={0: 1.0, 1: 1.0},
        )

        # 每张卡先拿到第一个 slot，避免启动阶段一张卡独占所有初始化资源。
        self.assertEqual([slot.gpu_id for slot in pool.slots], [0, 1, 0, 1])

    def test_tester_owns_process_preparation_entry(self):
        # engine 只应调用 tester 的统一入口，不直接识别具体 backend。
        import tester
        from tester.input_generation.backend import resolve_input_backend_policy

        prepare_process_runtime = getattr(tester, "prepare_process_runtime", None)
        self.assertIsNotNone(prepare_process_runtime)
        policy = resolve_input_backend_policy(
            requested="numpy",
            use_gpu_mode=False,
            mode="paddle_only",
        )
        options = SimpleNamespace(runtime_config=SimpleNamespace(input_backend_policy=policy))
        self.assertEqual(prepare_process_runtime(options).name, "numpy")


if __name__ == "__main__":
    unittest.main()
