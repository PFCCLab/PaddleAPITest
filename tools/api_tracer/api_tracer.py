from __future__ import annotations

import os
import signal
import sys
import warnings
from typing import Literal, TypedDict

try:
    from typing import Unpack
except ImportError:
    from typing import Unpack

from .config_serializer import ConfigSerializer
from .framework_dialect import FrameworkDialect, TracingHook


class APITracerKwargs(TypedDict, total=False):
    merge_output: bool
    record_stack: bool
    stack_format: Literal["full", "short", "api"]
    disable_torch_api_list: bool


class APITracer:
    def __init__(
        self,
        dialect: str,
        output_path: str = "trace_output",
        levels: int | list = 0,
        **kwargs: Unpack[APITracerKwargs],
    ):
        """初始化 API 追踪器

        Args:
        - dialect (str): 指定抓取的框架方言, 例如 "torch"
        - output_path (str, optional): 输出文件的路径, 默认为 "trace_output"
        - levels (Union[int, List], optional): 抓取配置的粒度, 可以是单个整数或整数列表, 默认为 0, 可选项有: 0, 1, 2

        Kwargs:
        - merge_output (bool, optional): 是否合并输出, 默认为 False
        - record_stack (bool, optional): 是否记录堆栈信息, 默认为 False
        - stack_format (str, optional): 堆栈信息的格式, 默认为 "short", 可选值有: "full", "short", "api"
        - disable_torch_api_list (bool, optional): 是否禁用 Torch API 列表, 默认为 False
        """
        if invalid := set(kwargs) - set(APITracerKwargs.__annotations__):
            raise ValueError(f"Invalid keyword arguments: {sorted(invalid)}")

        self.record_stack = kwargs.get("record_stack", False)
        if "stack_format" in kwargs:
            stack_format = kwargs.get("stack_format")
            if self.record_stack and stack_format not in ["full", "short", "api"]:
                raise ValueError(
                    f"Invalid stack_format: {stack_format}, it should be one of ['full', 'short', 'api']"
                )

        self.merge_output = kwargs.get("merge_output", False)
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        levels = levels if isinstance(levels, list) else [levels]

        self.dialect = FrameworkDialect.get_dialect(dialect)
        self.serializer = ConfigSerializer(self.dialect, output_path, levels, **kwargs)
        self.hooks: list[TracingHook] = self.dialect.get_hooks(self.serializer, levels, **kwargs)
        self._is_tracing = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        print(
            f"[APITracer] API tracer initialized for '{self.dialect.get_framework_name()}' "
            f"in levels {levels}, output path: {output_path}"
        )
        print(f"[APITracer] Kwargs: {kwargs}")

    def _signal_handler(self, signum, frame):
        print(f"[APITracer] Received signal {signum}, stopping trace...")
        self.stop()
        sys.exit(1)

    def start(self):
        """启动抓取"""
        if self._is_tracing:
            print("[APITracer] Tracing is already active.")
            return

        print("[APITracer] Starting API trace...")
        self.serializer.open()
        for hook in self.hooks:
            hook.install()
        self._is_tracing = True
        print("[APITracer] Tracing is now ACTIVE.")

    def stop(self):
        """停止抓取并恢复"""
        if not self._is_tracing:
            print("[APITracer] Tracing is not active.")
            return

        print("[APITracer] Stopping API trace...")
        for hook in reversed(self.hooks):
            hook.uninstall()
        self.serializer.close()
        self._is_tracing = False
        print("[APITracer] Tracing stopped and all APIs have been restored.")
        self.serializer.get_apis_and_configs()
        if self.record_stack:
            self.serializer.get_api_stacks()
        if self.merge_output:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "api_list",
                "torch_api_list.yaml",
            )
            self.get_alias_apis(self.output_path, yaml_path)

    def __enter__(self):
        """进入上下文管理器"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文管理器"""
        self.stop()

    @staticmethod
    def parse_configs_from_trace(input_path: str, output_path: str, output_suffix: str = ""):
        """从 api_trace.txt 文件中解析 API 配置和统计信息

        等价于 ConfigSerializer.parse_trace_configs
        """
        ConfigSerializer.parse_trace_configs(input_path, output_path, output_suffix)

    @staticmethod
    def parse_stacks_from_trace(input_path: str, output_path: str, output_suffix: str = ""):
        """从 api_trace.yaml 文件中解析每个 API 的前 3 个调用堆栈

        等价于 ConfigSerializer.parse_trace_stacks
        """
        ConfigSerializer.parse_trace_stacks(input_path, output_path, output_suffix)

    @staticmethod
    def get_alias_apis(
        input_path: str,
        yaml_path: str,
    ):
        """解析 api_apis.txt 文件, 生成 alias_api.txt 和 excluded_api.txt

        等价于 api_alias.get_alias_apis
        """
        from .api_alias_tool import get_alias_apis

        get_alias_apis(input_path, yaml_path)

    @staticmethod
    def get_merged_model_apis(
        input_path: str,
        output_path: str,
        sheet_name: str | None = None,
        model_groups: dict[str, list[str]] | None = None,
        yaml_paths: dict[str, str] | None = None,
    ):
        """从 XLSX 或 CSV 文件中读取 API 数据, 合并并分析 API

        等价于 api_merge.merge_model_apis
        """
        from .api_merge_tool import get_merged_model_apis

        get_merged_model_apis(input_path, output_path, sheet_name, model_groups, yaml_paths)

    @staticmethod
    def get_mapped_model_apis(
        torch_static_path: str,
        torch_dynamic_path: str,
        paddle_dynamic_path: str,
        mapping_table_path: str,
        output_path: str,
    ):
        """根据给定的 API 列表和映射表文件, 生成详细的 API 映射报告

        等价于 api_map_tool.get_mapped_model_apis
        """
        from .api_map_tool import get_mapped_model_apis

        get_mapped_model_apis(
            torch_static_path,
            torch_dynamic_path,
            paddle_dynamic_path,
            mapping_table_path,
            output_path,
        )
