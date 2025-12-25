"""单测文件生成器模块"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _generate_tensor_code_using_get_numpy_tensor(
    tensor_config, var_name: str, api_config_var: str, index: int = None, key: str = None
) -> str:
    """生成使用TensorConfig.get_numpy_tensor()的Python代码，与paddle_device_vs_cpu.py中的逻辑一致"""
    shape_str = str(tensor_config.shape)
    dtype_str = tensor_config.dtype
    if index is not None:
        return (
            f"# 生成tensor: shape={shape_str}, dtype={dtype_str}\n"
            f"{var_name}_cfg = TensorConfig({shape_str}, '{dtype_str}')\n"
            f"{var_name}_cfg.get_numpy_tensor({api_config_var}, index={index})\n"
            f"{var_name} = {var_name}_cfg.numpy_tensor"
        )
    elif key is not None:
        return (
            f"# 生成tensor: shape={shape_str}, dtype={dtype_str}\n"
            f"{var_name}_cfg = TensorConfig({shape_str}, '{dtype_str}')\n"
            f"{var_name}_cfg.get_numpy_tensor({api_config_var}, key='{key}')\n"
            f"{var_name} = {var_name}_cfg.numpy_tensor"
        )
    else:
        # 如果没有index和key，使用get_random_numpy_tensor作为fallback
        return (
            f"# 生成随机tensor: shape={shape_str}, dtype={dtype_str}\n"
            f"{var_name}_cfg = TensorConfig({shape_str}, '{dtype_str}')\n"
            f"{var_name} = {var_name}_cfg.get_random_numpy_tensor(shape={shape_str}, data_type='{dtype_str}')"
        )


def _extract_tensor_config_from_item(config_item):
    """提取TensorConfig信息"""
    from .api_config.config_analyzer import TensorConfig

    if isinstance(config_item, TensorConfig):
        return config_item
    elif isinstance(config_item, (list, tuple)):
        result = []
        for item in config_item:
            extracted = _extract_tensor_config_from_item(item)
            if extracted is not None:
                result.append(extracted)
        return tuple(result) if isinstance(config_item, tuple) else result
    return None


def _extract_tensor_configs(
    api_config, args_config, kwargs_config
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """从API配置中提取所有TensorConfig信息"""
    from .api_config.config_analyzer import TensorConfig

    args_configs = []
    kwargs_configs = {}

    def extract_from_config(config_item, index=None, key=None):
        if isinstance(config_item, TensorConfig):
            return config_item
        elif isinstance(config_item, (list, tuple)):
            result = []
            for i, item in enumerate(config_item):
                extracted = extract_from_config(item, index=i)
                if extracted is not None:
                    result.append(extracted)
            return tuple(result) if isinstance(config_item, tuple) else result
        return None

    for i, arg_config in enumerate(args_config):
        tensor_config = extract_from_config(arg_config, index=i)
        if tensor_config is not None:
            args_configs.append((f"arg_{i}", tensor_config))

    for key, kwarg_config in kwargs_config.items():
        tensor_config = extract_from_config(kwarg_config, key=key)
        if tensor_config is not None:
            kwargs_configs[key] = tensor_config

    return args_configs, kwargs_configs


def _generate_test_code(
    api_name: str,
    api_config_str: str,
    args_configs: list[tuple[str, Any]],
    kwargs_configs: dict[str, Any],
    error_info: dict[str, Any],
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
    non_tensor_args: list[tuple[int, Any]] = None,
    non_tensor_kwargs: dict[str, Any] = None,
) -> str:
    """生成单测文件代码"""
    code_lines = [
        "import sys",
        "import os",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        "",
        '"""',
        f"自动生成的单测文件 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"API: {api_name}",
        (
            f"配置: {api_config_str[:200]}..."
            if len(api_config_str) > 200
            else f"配置: {api_config_str}"
        ),
        f"错误类型: {error_info.get('error_type', 'unknown')}",
        f"失败阶段: {error_info.get('stage', 'unknown')}",
        '"""',
        "",
        "import paddle",
        "import numpy",
        "import torch",
        "from tester.api_config.config_analyzer import TensorConfig, APIConfig",
        "",
        f"# 创建API配置对象",
        f'api_config = APIConfig("{api_config_str}")',
        "",
        f"# 设置目标设备",
    ]
    
    # 修复CPU设备设置：如果是cpu，不要加:0
    if target_device == "cpu":
        code_lines.append('paddle.set_device("cpu")')
    else:
        code_lines.append(f'paddle.set_device("{target_device}:{device_id}")')
    code_lines.append("")

    is_tensor_method = api_name.startswith("paddle.Tensor.")
    if is_tensor_method and not args_configs and kwargs_configs:
        first_key = next(iter(kwargs_configs.keys()))
        first_value = kwargs_configs.pop(first_key)
        args_configs.insert(0, (first_key, first_value))

    code_lines.append("# 生成输入数据")
    code_lines.append("numpy.random.seed(0)")
    code_lines.append("")
    all_inputs = []
    tensor_vars = {}

    from .api_config.config_analyzer import TensorConfig

    # 使用TensorConfig.get_numpy_tensor()方法生成数据，与paddle_device_vs_cpu.py中的逻辑一致
    for i, (var_name, tensor_config) in enumerate(args_configs):
        if isinstance(tensor_config, (list, tuple)):
            code_lines.append(f"# 位置参数 {var_name} (list/tuple)")
            tensor_list = []
            for j, item in enumerate(tensor_config):
                if isinstance(item, TensorConfig):
                    item_var = f"{var_name}_item_{j}"
                    code_lines.append(
                        _generate_tensor_code_using_get_numpy_tensor(
                            item, item_var, "api_config", index=i
                        )
                    )
                    tensor_var = f"{item_var}_tensor"
                    code_lines.append(f"{tensor_var} = paddle.to_tensor({item_var})")
                    tensor_list.append(tensor_var)
                    all_inputs.append(tensor_var)
            if tensor_list:
                code_lines.append(f"{var_name}_tensors = [{', '.join(tensor_list)}]")
        elif isinstance(tensor_config, TensorConfig):
            code_lines.append(f"# 位置参数 {var_name}")
            code_lines.append(
                _generate_tensor_code_using_get_numpy_tensor(
                    tensor_config, var_name, "api_config", index=i
                )
            )
            tensor_var = f"{var_name}_tensor"
            code_lines.append(f"{tensor_var} = paddle.to_tensor({var_name})")
            tensor_vars[var_name] = tensor_var
            all_inputs.append(tensor_var)

    # 处理kwargs，使用TensorConfig.get_numpy_tensor()方法
    for key, tensor_config in kwargs_configs.items():
        if isinstance(tensor_config, (list, tuple)):
            code_lines.append(f"# 关键字参数 {key} (list/tuple)")
            tensor_list = []
            for j, item in enumerate(tensor_config):
                if isinstance(item, TensorConfig):
                    item_var = f"kwarg_{key}_item_{j}"
                    code_lines.append(
                        _generate_tensor_code_using_get_numpy_tensor(
                            item, item_var, "api_config", key=key
                        )
                    )
                    tensor_var = f"{item_var}_tensor"
                    code_lines.append(f"{tensor_var} = paddle.to_tensor({item_var})")
                    tensor_list.append(tensor_var)
                    all_inputs.append(tensor_var)
            if tensor_list:
                code_lines.append(f"kwarg_{key}_tensors = [{', '.join(tensor_list)}]")
        elif isinstance(tensor_config, TensorConfig):
            code_lines.append(f"# 关键字参数 {key}")
            var_name = f"kwarg_{key}"
            code_lines.append(
                _generate_tensor_code_using_get_numpy_tensor(
                    tensor_config, var_name, "api_config", key=key
                )
            )
            tensor_var = f"{var_name}_tensor"
            code_lines.append(f"{tensor_var} = paddle.to_tensor({var_name})")
            tensor_vars[key] = tensor_var
            all_inputs.append(tensor_var)

    code_lines.append("")
    code_lines.append("# 构建API调用参数")

    if non_tensor_args is None:
        non_tensor_args = []
    if non_tensor_kwargs is None:
        non_tensor_kwargs = {}

    for idx, value in non_tensor_args:
        code_lines.append(f"# 非tensor位置参数 arg_{idx}")
        if isinstance(value, str):
            code_lines.append(f'arg_{idx}_non_tensor = "{value}"')
        else:
            code_lines.append(f"arg_{idx}_non_tensor = {repr(value)}")

    for key, value in non_tensor_kwargs.items():
        code_lines.append(f"# 非tensor关键字参数 {key}")
        if isinstance(value, str):
            code_lines.append(f'kwarg_{key}_non_tensor = "{value}"')
        else:
            code_lines.append(f"kwarg_{key}_non_tensor = {repr(value)}")

    code_lines.append("")

    arg_vars = []
    # 使用配置方式构建arg_vars
    config_idx = 0
    non_tensor_idx = 0
    max_args = max(len(args_configs) + len(non_tensor_args), 0)
    for i in range(max_args):
        if config_idx < len(args_configs) and args_configs[config_idx][0] == f"arg_{i}":
            var_name, tensor_config = args_configs[config_idx]
            if isinstance(tensor_config, (list, tuple)):
                arg_vars.append(f"{var_name}_tensors")
            else:
                arg_vars.append(tensor_vars.get(var_name, var_name))
            config_idx += 1
        elif non_tensor_idx < len(non_tensor_args) and non_tensor_args[non_tensor_idx][0] == i:
            arg_vars.append(f"arg_{i}_non_tensor")
            non_tensor_idx += 1

    kwarg_vars = {}
    for key, tensor_config in kwargs_configs.items():
        if isinstance(tensor_config, (list, tuple)):
            kwarg_vars[key] = f"kwarg_{key}_tensors"
        else:
            kwarg_vars[key] = tensor_vars.get(key, f"kwarg_{key}_tensor")

    for key in non_tensor_kwargs:
        kwarg_vars[key] = f"kwarg_{key}_non_tensor"

    code_lines.append("")
    code_lines.append("# 执行API调用")
    code_lines.append("try:")

    is_tensor_method = api_name.startswith("paddle.Tensor.")
    if is_tensor_method:
        method_name = api_name.split(".")[-1]
        if arg_vars:
            tensor_var = arg_vars[0]
            remaining_args = arg_vars[1:] if len(arg_vars) > 1 else []
        else:
            tensor_var = None
            remaining_args = []
    else:
        tensor_var = None
        remaining_args = arg_vars

    api_call_parts = []
    if not is_tensor_method:
        if remaining_args:
            api_call_parts.extend(remaining_args)
    else:
        if remaining_args:
            api_call_parts.extend(remaining_args)

    if kwarg_vars:
        kwarg_str = ", ".join([f"{k}={v}" for k, v in kwarg_vars.items()])
        api_call_parts.append(kwarg_str)

    if is_tensor_method and tensor_var:
        if api_call_parts:
            api_call = f"    output = {tensor_var}.{method_name}(" + ", ".join(api_call_parts) + ")"
        else:
            api_call = f"    output = {tensor_var}.{method_name}()"
    else:
        if api_call_parts:
            api_call = f"    output = {api_name}(" + ", ".join(api_call_parts) + ")"
        else:
            api_call = f"    output = {api_name}()"

    if test_amp:
        code_lines.append("    with paddle.amp.auto_cast():")
        code_lines.append("        " + api_call)
    else:
        code_lines.append(api_call)

    code_lines.append('    print("Forward pass succeeded")')
    code_lines.append('    print(f"Output type: {type(output)}")')
    code_lines.append("    if isinstance(output, paddle.Tensor):")
    code_lines.append('        print(f"Output shape: {output.shape}, dtype: {output.dtype}")')
    code_lines.append("    elif isinstance(output, (list, tuple)):")
    code_lines.append('        print(f"Output length: {len(output)}")')
    code_lines.append("        for i, item in enumerate(output):")
    code_lines.append("            if isinstance(item, paddle.Tensor):")
    code_lines.append(
        '                print(f"  Output[{i}]: shape={item.shape}, dtype={item.dtype}")'
    )
    code_lines.append("")

    if error_info.get("stage") == "backward" or error_info.get("need_backward", False):
        code_lines.append("")
        code_lines.append("    # Backward测试")
        code_lines.append("    if isinstance(output, paddle.Tensor):")
        code_lines.append("        output.backward()")
        code_lines.append("    elif isinstance(output, (list, tuple)):")
        code_lines.append("        for item in output:")
        code_lines.append("            if isinstance(item, paddle.Tensor):")
        code_lines.append("                item.backward()")
        code_lines.append('    print("Backward pass succeeded")')

    code_lines.append("except Exception as e:")
    code_lines.append('    print(f"Error occurred: {e}")')
    code_lines.append("    import traceback")
    code_lines.append("    traceback.print_exc()")
    code_lines.append("    raise")

    return "\n".join(code_lines)


def generate_reproducible_test_file(
    api_config,
    error_info: dict[str, Any],
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
    test_instance=None,
) -> str | None:
    """生成可复现的单测文件"""
    try:
        output_dir = "failed_tests"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        args_configs = []
        kwargs_configs = {}
        non_tensor_args = []
        non_tensor_kwargs = {}

        if test_instance is not None:
            if hasattr(test_instance, "paddle_args_config"):
                for i, arg_config in enumerate(test_instance.paddle_args_config):
                    tensor_config = _extract_tensor_config_from_item(arg_config)
                    if tensor_config is not None:
                        args_configs.append((f"arg_{i}", tensor_config))
                    else:
                        from .api_config.config_analyzer import TensorConfig

                        if not isinstance(arg_config, TensorConfig):
                            non_tensor_args.append((i, arg_config))

            if hasattr(test_instance, "paddle_kwargs_config"):
                for key, kwarg_config in test_instance.paddle_kwargs_config.items():
                    tensor_config = _extract_tensor_config_from_item(kwarg_config)
                    if tensor_config is not None:
                        kwargs_configs[key] = tensor_config
                    else:
                        from .api_config.config_analyzer import TensorConfig

                        if not isinstance(kwarg_config, TensorConfig):
                            non_tensor_kwargs[key] = kwarg_config

        # 不再提取实际数据，而是使用TensorConfig配置来生成数据
        # 这样可以确保使用和paddle_device_vs_cpu.py中相同的生成逻辑

        if not args_configs and not kwargs_configs:
            args_configs, kwargs_configs = _extract_tensor_configs(
                api_config,
                api_config.args if hasattr(api_config, "args") else [],
                api_config.kwargs if hasattr(api_config, "kwargs") else {},
            )

        api_name_safe = api_config.api_name.replace(".", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(api_config.config.encode()).hexdigest()[:8]
        pid = os.getpid()
        filename = f"test_{api_name_safe}_{timestamp}_{pid}_{config_hash}.py"
        filepath = output_path / filename

        test_code = _generate_test_code(
            api_config.api_name,
            api_config.config,
            args_configs,
            kwargs_configs,
            error_info,
            test_amp=test_amp,
            target_device=target_device,
            device_id=device_id,
            non_tensor_args=non_tensor_args,
            non_tensor_kwargs=non_tensor_kwargs,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(test_code)

        print(f"[Generated test file] {filepath}", flush=True)
        return str(filepath)

    except Exception as e:
        print(f"[Error generating test file] {e}", flush=True)
        import traceback

        traceback.print_exc()
        return None
