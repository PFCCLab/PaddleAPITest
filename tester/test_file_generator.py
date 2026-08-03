"""单测文件生成器模块"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _device_setup_lines(target_device: str, device_id: int) -> list[str]:
    if target_device == "cpu":
        return ['paddle.set_device("cpu")']
    return [f'paddle.set_device("{target_device}:{device_id}")']


def _generated_header(api_name: str, api_config_str: str, error_info: dict[str, Any]) -> list[str]:
    config_text = (
        f"配置: {api_config_str[:200]}..."
        if len(api_config_str) > 200
        else f"配置: {api_config_str}"
    )
    return [
        "import sys",
        "import os",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
        "",
        '"""',
        f"自动生成的单测文件 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"API: {api_name}",
        config_text,
        f"错误类型: {error_info.get('error_type', 'unknown')}",
        f"失败阶段: {error_info.get('stage', 'unknown')}",
        '"""',
        "",
        "import paddle",
        "from tester.api_config.parser import APIConfig",
        "",
        f"api_config = APIConfig({api_config_str!r})",
        "",
    ]


def _accuracy_repro_lines(
    test_amp: bool,
    target_device: str,
    device_id: int,
) -> list[str]:
    lines = [
        "# 使用 APITestCustomDeviceVSCPU 运行 CPU 与目标设备对比测试",
        "from tester.paddle_device_vs_cpu import APITestCustomDeviceVSCPU",
        "",
        "test_instance = APITestCustomDeviceVSCPU(",
        "    api_config,",
        f"    test_amp={test_amp},",
    ]
    if target_device == "xpu":
        lines.append(f"    xpu_device_id={device_id},")
    lines.extend(
        [
            "    generate_failed_tests=False,",
            ")",
            "",
            "test_instance.test()",
            "print('[Test completed]', flush=True)",
        ]
    )
    return lines


def _paddle_only_repro_lines(error_info: dict[str, Any], test_amp: bool) -> list[str]:
    lines = [
        "# 使用当前 input_generation 主流程生成输入并构造 Paddle 调用参数",
        "from tester.base import APITestBase",
        "",
        f"test_base = APITestBase(api_config, use_torch=False)",
        "if not test_base.ana_paddle_api_info():",
        "    raise RuntimeError('ana_paddle_api_info failed')",
        "if not test_base.gen_input_data():",
        "    raise RuntimeError('gen_input_data failed')",
        "if not test_base.build_paddle_input():",
        "    raise RuntimeError('build_paddle_input failed')",
        "",
        "try:",
    ]
    if test_amp:
        lines.extend(
            [
                "    with paddle.amp.auto_cast():",
                "        output = test_base.paddle_api(",
                "            *tuple(test_base.paddle_args),",
                "            **test_base.paddle_kwargs,",
                "        )",
            ]
        )
    else:
        lines.extend(
            [
                "    output = test_base.paddle_api(",
                "        *tuple(test_base.paddle_args),",
                "        **test_base.paddle_kwargs,",
                "    )",
            ]
        )

    lines.extend(
        [
            '    print("Forward pass succeeded")',
            '    print(f"Output type: {type(output)}")',
            "    if isinstance(output, paddle.Tensor):",
            '        print(f"Output shape: {output.shape}, dtype={output.dtype}")',
            "    elif isinstance(output, (list, tuple)):",
            '        print(f"Output length: {len(output)}")',
            "        for i, item in enumerate(output):",
            "            if isinstance(item, paddle.Tensor):",
            '                print(f"  Output[{i}]: shape={item.shape}, dtype={item.dtype}")',
        ]
    )

    if error_info.get("stage") == "backward" or error_info.get("need_backward", False):
        lines.extend(
            [
                "",
                "    if isinstance(output, paddle.Tensor):",
                "        output.backward()",
                "    elif isinstance(output, (list, tuple)):",
                "        for item in output:",
                "            if isinstance(item, paddle.Tensor):",
                "                item.backward()",
                '    print("Backward pass succeeded")',
            ]
        )

    lines.extend(
        [
            "finally:",
            "    test_base.clear_runtime_inputs('paddle')",
        ]
    )
    return lines


def _generate_test_code(
    api_name: str,
    api_config_str: str,
    error_info: dict[str, Any],
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
) -> str:
    """生成可复现的单测文件代码。"""
    is_accuracy_error = error_info.get("error_type") == "accuracy_error"
    code_lines = _generated_header(api_name, api_config_str, error_info)

    if not is_accuracy_error:
        code_lines.extend(["# 设置目标设备", *_device_setup_lines(target_device, device_id), ""])

    if is_accuracy_error:
        code_lines.extend(_accuracy_repro_lines(test_amp, target_device, device_id))
    else:
        code_lines.extend(_paddle_only_repro_lines(error_info, test_amp))

    return "\n".join(code_lines)


def generate_reproducible_test_file(
    api_config,
    error_info: dict[str, Any],
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
    test_instance=None,
) -> str | None:
    """生成可复现的单测文件。"""
    try:
        output_path = Path("failed_tests")
        output_path.mkdir(parents=True, exist_ok=True)

        api_name_safe = api_config.api_name.replace(".", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(api_config.config.encode()).hexdigest()[:8]
        filename = f"test_{api_name_safe}_{timestamp}_{os.getpid()}_{config_hash}.py"
        filepath = output_path / filename

        test_code = _generate_test_code(
            api_config.api_name,
            api_config.config,
            error_info,
            test_amp=test_amp,
            target_device=target_device,
            device_id=device_id,
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
