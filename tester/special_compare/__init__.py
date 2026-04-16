"""
device_vs_gpu 特殊对比注册表。

每个 API 的特殊对比逻辑放在本目录下独立的文件中（如 argsort.py）。

添加新 API 只需：
  1. 在本目录新建一个文件，如 topk.py
  2. 在文件中编写对比函数并用装饰器注册
  3. 不需要修改任何其他文件，系统会自动发现新文件

对比函数签名：
  compare_forward(local_output, remote_output, api_config, tester) -> None
  compare_backward(local_grads, remote_grads, api_config, tester) -> None
  失败时抛出 AssertionError（与 np.testing.assert_allclose 行为一致），成功返回 None。

参数说明：
  local_output / local_grads:  XPU 侧的输出/梯度（paddle.Tensor 或 list[Tensor]）
  remote_output / remote_grads: GPU 侧的输出/梯度（从 BOS 下载并 paddle.load 还原）
  api_config:  APIConfig 实例，提供 api_name / args / kwargs
  tester:      APITestPaddleDeviceVSGPU 实例，提供：
               - tester.paddle_args[0]  原始输入张量（XPU 侧，与 GPU 侧同数据）
               - tester.atol / tester.rtol  命令行配置的精度容差
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_FORWARD_REGISTRY: dict[str, callable] = {}
_BACKWARD_REGISTRY: dict[str, callable] = {}


class SkipComparison(Exception):
    """
    在特殊对比函数中抛出此异常，表示跳过本次对比，结果记为 skip。

    用法：
        @register_forward("paddle.some_api")
        def compare_some_api(local_output, remote_output, api_config, tester):
            raise SkipComparison("XPU 暂不支持该 API 的精度对比")

    或直接使用快捷装饰器：
        @register_skip("paddle.some_api", "paddle.Tensor.some_api")
        def _skip_some_api(): ...   # 函数体不重要，永远不会被执行
    """


def register_forward(*api_names: str):
    """
    装饰器：注册一个前向（Forward output）特殊对比函数。

    可同时注册多个 API 别名到同一个函数：
        @register_forward("paddle.argsort", "paddle.Tensor.argsort")
        def compare_argsort_forward(local_output, remote_output, api_config, tester):
            ...
    """
    def decorator(fn):
        for name in api_names:
            if name in _FORWARD_REGISTRY:
                raise ValueError(
                    f"register_forward: '{name}' 已注册到 {_FORWARD_REGISTRY[name].__name__!r}，"
                    "请勿重复注册"
                )
            _FORWARD_REGISTRY[name] = fn
        return fn
    return decorator


def register_backward(*api_names: str):
    """
    装饰器：注册一个反向（Backward gradient）特殊对比函数。

    可同时注册多个 API 别名到同一个函数：
        @register_backward("paddle.some_api", "paddle.Tensor.some_api")
        def compare_some_api_backward(local_grads, remote_grads, api_config, tester):
            ...
    """
    def decorator(fn):
        for name in api_names:
            if name in _BACKWARD_REGISTRY:
                raise ValueError(
                    f"register_backward: '{name}' 已注册到 {_BACKWARD_REGISTRY[name].__name__!r}，"
                    "请勿重复注册"
                )
            _BACKWARD_REGISTRY[name] = fn
        return fn
    return decorator


def get_forward_compare(api_name: str):
    """
    返回 api_name 已注册的前向对比函数，未注册则返回 None。
    调用方应检查返回值，为 None 时回退到默认对比逻辑。
    """
    return _FORWARD_REGISTRY.get(api_name, None)


def get_backward_compare(api_name: str):
    """
    返回 api_name 已注册的反向对比函数，未注册则返回 None。
    调用方应检查返回值，为 None 时回退到默认对比逻辑。
    """
    return _BACKWARD_REGISTRY.get(api_name, None)


def register_skip(*api_names: str):
    """
    快捷装饰器：将 API 的 forward 和 backward 对比均标记为跳过。
    等价于同时注册 forward 和 backward 函数，两者都抛出 SkipComparison。

    用法：
        @register_skip("paddle.some_api", "paddle.Tensor.some_api")
        def _(): ...   # 函数体不会被执行，名字不重要
    """
    def decorator(fn):
        def _skip(*, reason=""):
            raise SkipComparison(reason or f"{api_names[0]} 已注册为跳过对比")

        for name in api_names:
            if name in _FORWARD_REGISTRY:
                raise ValueError(
                    f"register_skip: '{name}' 已注册到 {_FORWARD_REGISTRY[name].__name__!r}"
                )
            if name in _BACKWARD_REGISTRY:
                raise ValueError(
                    f"register_skip: '{name}' 已注册到 {_BACKWARD_REGISTRY[name].__name__!r}"
                )

            def _skip_fn(*args, _name=name, **kwargs):
                raise SkipComparison(f"{_name} 已注册为跳过对比")

            _FORWARD_REGISTRY[name] = _skip_fn
            _BACKWARD_REGISTRY[name] = _skip_fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# 自动发现并导入本目录下所有子模块
# 每个子模块文件顶层的 @register_forward / @register_backward 装饰器会在 import
# 时立即执行，完成注册。新增 API 文件只需放入本目录，无需修改此文件。
# ---------------------------------------------------------------------------
_pkg_dir = Path(__file__).parent
for _mod_info in pkgutil.iter_modules([str(_pkg_dir)]):
    importlib.import_module(f".{_mod_info.name}", package=__name__)
