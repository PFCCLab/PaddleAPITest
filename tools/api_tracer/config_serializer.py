from __future__ import annotations

import itertools
import linecache
import os
import queue
import sys
from collections import defaultdict
from threading import Event, Thread
from typing import Any, TextIO

import yaml

from .framework_dialect import FrameworkDialect


class ConfigSerializer:
    """将API调用信息序列化并写入"""

    def __init__(
        self,
        dialect: FrameworkDialect,
        output_path: str,
        levels: list[int],
        **kwargs: Any,
    ):
        self.dialect = dialect
        self.output_path = output_path
        self.levels = levels
        self.merge_output = kwargs.get("merge_output", False)
        self.record_stack = kwargs.get("record_stack", False)
        self.stack_format = kwargs.get("stack_format", "short")

        self.file_handlers: dict[int, dict[str, TextIO]] = {}
        self.buffer_limit = 20000
        self.buffers: dict[int, list[dict]] = defaultdict(list)

        self.max_args_count = 100
        self.max_item_count = 100
        self.max_line_length = 1024
        self.max_nest_depth = 5

        self._call_counter = itertools.count()
        self._excluded_record_files = (
            "framework_dialect.py",
            "eval_frame.py",
            "module.py",
        )

        # asyncio
        self.log_queue = queue.Queue()
        self._stop_event = Event()
        self.writer_thread = Thread(target=self._writer_loop)
        self.total_calls_processed = 0

        self._serialize_handlers = {
            type(None): lambda x, depth: x,
            bool: lambda x, depth: x,
            int: lambda x, depth: x,
            float: lambda x, depth: x,
            str: lambda x, depth: x,
            list: self._serialize_list,
            tuple: self._serialize_tuple,
            set: self._serialize_set,
            dict: self._serialize_dict,
            type: self._serialize_type,
            slice: self._serialize_slice,
            type(Ellipsis): self._serialize_ellipsis,
        }

    def open(self):
        if self.merge_output:
            self.file_handlers[0] = {
                "yaml": open(f"{self.output_path}/api_trace.yaml", "w", encoding="utf-8"),
                "txt": open(f"{self.output_path}/api_trace.txt", "w", encoding="utf-8"),
            }
        else:
            for level in self.levels:
                self.file_handlers[level] = {
                    "yaml": open(
                        f"{self.output_path}/api_trace_level_{level}.yaml",
                        "w",
                        encoding="utf-8",
                    ),
                    "txt": open(
                        f"{self.output_path}/api_trace_level_{level}.txt",
                        "w",
                        encoding="utf-8",
                    ),
                }
        self.writer_thread.start()

    def close(self):
        self._stop_event.set()
        self.writer_thread.join()
        for level in self.buffers:
            self._flush_buffer(level)

        for level_handlers in self.file_handlers.values():
            if level_handlers["yaml"]:
                level_handlers["yaml"].close()
            if level_handlers["txt"]:
                level_handlers["txt"].close()

        self.file_handlers.clear()
        print(f"[ConfigSerializer] Files closed, final save to {self.output_path}")

    def _writer_loop(self):
        """线程的工作循环，从队列获取数据并处理"""
        while not self._stop_event.is_set():
            try:
                call_record = self.log_queue.get(timeout=0.1)

                if self.record_stack:
                    frames_data = call_record.pop("frames_data")
                    if self.stack_format == "api":
                        api_stack = [self._format_frame_to_api(f) for f in frames_data]
                    elif self.stack_format == "full":
                        api_stack = [self._format_frame_to_traceback(f) for f in frames_data]
                    else:  # "short" format
                        api_stack = [self._format_frame_to_short(f) for f in frames_data]
                    call_record["stack"] = api_stack

                level = call_record.pop("level")
                buffer_key = 0 if self.merge_output else level
                self.buffers[buffer_key].append(call_record)

                if len(self.buffers[buffer_key]) >= self.buffer_limit:
                    self._flush_buffer(buffer_key)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ConfigSerializer] Error in writer thread: {e}")

    def _flush_buffer(self, buffer_key: int):
        """将buffer内容写入文件并清空buffer"""
        buffer = self.buffers[buffer_key]
        if not buffer:
            return

        self.total_calls_processed += len(buffer)
        handlers = self.file_handlers[buffer_key]

        if handlers["yaml"]:
            try:
                yaml.dump_all(
                    buffer,
                    handlers["yaml"],
                    Dumper=yaml.CDumper,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )
                handlers["yaml"].flush()
            except Exception as e:
                print(f"[ConfigSerializer] Error writing YAML file: {e}")

        if handlers["txt"]:
            try:
                for call_record in buffer:
                    txt_line = self._format_txt_line(
                        call_record["api"], call_record["args"], call_record["kwargs"]
                    )
                    handlers["txt"].write(txt_line + "\n")
                handlers["txt"].flush()
            except Exception as e:
                print(f"[ConfigSerializer] Error writing TXT file: {e}")

        print(
            f"[ConfigSerializer] Flushed {len(buffer)} calls for level key {buffer_key}, total processed: {self.total_calls_processed}"
        )
        buffer.clear()

    def _format_frame_to_api(self, frame_info: dict) -> str:
        """将帧信息格式化为 API 名称"""
        filename = frame_info["filename"]
        module_name = frame_info["module_name"]
        class_name = frame_info["class_name"]
        func_name = frame_info["func_name"]

        if module_name:
            if module_name == "__main__":
                api_path = f"__main__.{os.path.basename(filename)}"
            else:
                api_path = module_name
            if class_name:
                api_path = f"{api_path}.{class_name}"
            return f"{api_path}.{func_name}"
        else:
            return f"<UnknownModule>.{os.path.basename(filename)}.{func_name}"

    def _format_frame_to_short(self, frame_info: dict) -> str:
        """将帧信息格式化为短字符串"""
        filename = frame_info["filename"]
        lineno = frame_info["lineno"]
        func_name = frame_info["func_name"]
        return f"{os.path.basename(filename)}:{lineno} in {func_name}"

    def _format_frame_to_traceback(self, frame_info: dict) -> str:
        """将帧信息格式化为 traceback 字符串"""
        filename = frame_info["filename"]
        lineno = frame_info["lineno"]
        func_name = frame_info["func_name"]

        file_info = f'  File "{filename}", line {lineno}, in {func_name}'
        try:
            source_code = linecache.getline(filename, lineno).strip()
        except Exception:
            source_code = ""
        if source_code:
            return f"{file_info}\n    {source_code}"
        return file_info

    def dump_call(
        self,
        api_name: str,
        args: tuple,
        kwargs: dict,
        output: Any = None,
        level: int = 0,
    ):
        """记录一次API调用"""
        try:
            total_args = len(args) + len(kwargs)
            if total_args > self.max_args_count:
                if len(args) < self.max_args_count:
                    kwargs = dict(list(kwargs.items())[: self.max_args_count - len(args) - 1])
                    kwargs["__truncated__"] = "<Truncated: max args exceeded>"
                else:
                    args = (
                        *list(args)[: self.max_args_count - 1],
                        "<Truncated: max args exceeded>",
                    )
                    kwargs = {}

            call_record = {
                "level": level,
                "api": api_name,
                "args": [self._serialize_item(arg, depth=0) for arg in args],
                "kwargs": {
                    key: self._serialize_item(value, depth=0) for key, value in kwargs.items()
                },
                # "output_summary": self._serialize_item(output, depth=0)
            }

            if self.record_stack:
                call_id = next(self._call_counter)
                call_record["id"] = call_id

                frames_data = []
                frame = sys._getframe(1)
                while frame:
                    if not frame.f_code.co_filename.endswith(self._excluded_record_files):
                        f_locals = frame.f_locals
                        class_name = None
                        if "self" in f_locals:
                            class_name = f_locals["self"].__class__.__name__
                        elif "cls" in f_locals:
                            class_name = f_locals["cls"].__name__

                        frames_data.append(
                            {
                                "filename": frame.f_code.co_filename,
                                "lineno": frame.f_lineno,
                                "func_name": frame.f_code.co_name,
                                "module_name": frame.f_globals.get("__name__"),
                                "class_name": class_name,
                            }
                        )
                    frame = frame.f_back
                call_record["frames_data"] = frames_data

            self.log_queue.put_nowait(call_record)
        except Exception as e:
            print(f"[ConfigSerializer] Error serializing call for '{api_name}': {e}")

    def _serialize_list(self, item: list, depth: int) -> dict:
        if len(item) > self.max_item_count:
            item = [*item[: self.max_item_count - 1], "<Truncated: max item count>"]
        return {
            "type": "list",
            "value": [self._serialize_item(sub_item, depth) for sub_item in item],
        }

    def _serialize_tuple(self, item: tuple, depth: int) -> dict:
        if len(item) > self.max_item_count:
            item = (*item[: self.max_item_count - 1], "<Truncated: max item count>")
        return {
            "type": "tuple",
            "value": [self._serialize_item(sub_item, depth) for sub_item in item],
        }

    def _serialize_set(self, item: set, depth: int) -> dict:
        if len(item) > self.max_item_count:
            item = set(list(item)[: self.max_item_count - 1])
        return {
            "type": "set",
            "value": [self._serialize_item(sub_item, depth) for sub_item in item],
        }

    def _serialize_dict(self, item: dict, depth: int) -> dict:
        if len(item) > self.max_item_count:
            item = dict(list(item.items())[: self.max_item_count - 1])
            item["__truncated__"] = "<Truncated: max item count>"
        return {
            "type": "dict",
            "value": {str(k): self._serialize_item(v, depth) for k, v in item.items()},
        }

    def _serialize_type(self, item: type, depth: int) -> dict:
        return {"type": "type", "value": f"{item.__module__}.{item.__name__}"}

    def _serialize_slice(self, item: slice, depth: int) -> dict:
        return {
            "type": "slice",
            "value": {"start": item.start, "stop": item.stop, "step": item.step},
        }

    def _serialize_ellipsis(self, item: Any, depth: int) -> dict:
        return {"type": "ellipsis", "value": "..."}

    def _serialize_item(self, item: Any, depth=0) -> Any:
        """递归序列化对象"""
        if depth > self.max_nest_depth:
            return "<Truncated: max depth exceeded>"

        handler = self._serialize_handlers.get(type(item))
        if handler:
            return handler(item, depth=depth + 1)

        special_serialization = self.dialect.serialize_special_type(item)
        if special_serialization is not None:
            return special_serialization

        try:
            return f"<Unserializable: {type(item).__name__}>"
        except Exception:
            return "<Unserializable: unknown type>"

    def _format_txt_line(self, api_name: str, args: list, kwargs: dict) -> str:
        """格式化API调用为最通用的TXT配置"""

        def format_arg(arg: Any) -> str:
            if arg is None or isinstance(arg, (bool, int, float)):
                return str(arg)
            if isinstance(arg, str):
                if len(arg) > 100:
                    return f'"{arg[:97]}..."'
                return f'"{arg}"'

            if isinstance(arg, dict) and "type" in arg:
                type_handlers = {
                    "list": lambda x: f"list[{', '.join(format_arg(item) for item in x['value'])}]",
                    "tuple": lambda x: f"tuple({', '.join(format_arg(item) for item in x['value'])})",
                    "set": lambda x: f"set({', '.join(format_arg(item) for item in x['value'])})",
                    "dict": lambda x: f"dict({', '.join(f'{k}={format_arg(v)}' for k, v in x['value'].items())})",
                    "type": lambda x: x["value"],
                    "slice": lambda x: f"slice({x['value']['start']}, {x['value']['stop']}, {x['value']['step']})",
                    "ellipsis": lambda x: "ellipsis(...)",
                }
                handler = type_handlers.get(arg["type"])
                if handler:
                    return handler(arg)

            special_format = self.dialect.format_special_type(arg)
            if special_format is not None:
                return special_format

            return str(arg)

        args_str = ", ".join(format_arg(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
        result = f"{api_name}({args_str + (', ' + kwargs_str if kwargs_str else '')})"

        if len(result) > self.max_line_length:
            result = result[: self.max_line_length - 4] + "...)"
        return result

    def get_apis_and_configs(self):
        if self.merge_output:
            input_path = os.path.join(self.output_path, "api_trace.txt")
            ConfigSerializer.parse_trace_configs(input_path, self.output_path, "")
        else:
            for level in self.levels:
                suffix = f"_level_{level}"
                input_path = os.path.join(self.output_path, f"api_trace{suffix}.txt")
                ConfigSerializer.parse_trace_configs(input_path, self.output_path, suffix)

    def get_api_stacks(self):
        if self.merge_output:
            input_path = os.path.join(self.output_path, "api_trace.yaml")
            ConfigSerializer.parse_trace_stacks(input_path, self.output_path, "")
        else:
            for level in self.levels:
                suffix = f"_level_{level}"
                input_path = os.path.join(self.output_path, f"api_trace{suffix}.yaml")
                ConfigSerializer.parse_trace_stacks(input_path, self.output_path, suffix)

    @staticmethod
    def parse_trace_configs(input_path: str, output_path: str, output_suffix: str = ""):
        print(f"[ConfigSerializer] Start to parse configs from: {input_path}")
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"[ConfigSerializer] Trace file not found, skipping: {input_path}")
            return

        if os.path.splitext(input_path)[1].lower() != ".txt":
            print(f"[ConfigSerializer] Trace file is not a txt file, skipping: {input_path}")
            return

        api_apis, api_configs = set(), set()
        api_counts = defaultdict(int)

        with open(input_path, encoding="utf-8") as f:
            current = ""
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if current:
                    current += " " + line
                    if line.endswith(")"):
                        api_configs.add(current)
                        current = ""
                    continue
                api_api = line.split("(", 1)[0]
                api_apis.add(api_api)
                api_counts[api_api] += 1
                if line.endswith(")"):
                    api_configs.add(line)
                else:
                    current = line

        trace_count = sum(api_counts.values())
        print(f"[ConfigSerializer] Read {trace_count} traces from {os.path.basename(input_path)}")

        with open(
            os.path.join(output_path, f"api_apis{output_suffix}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for api in sorted(api_apis):
                f.write(api + "\n")
        print(f"[ConfigSerializer] Write {len(api_apis)} apis to api_apis{output_suffix}.txt")

        with open(
            os.path.join(output_path, f"api_configs{output_suffix}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for config in sorted(api_configs):
                f.write(config + "\n")
        print(
            f"[ConfigSerializer] Write {len(api_configs)} configs to api_configs{output_suffix}.txt"
        )

        total_calls = sum(api_counts.values())
        if total_calls == 0:
            return
        api_percentages = {api: (count / total_calls) * 100 for api, count in api_counts.items()}
        sorted_api_counts = sorted(api_counts.items(), key=lambda x: x[1], reverse=True)

        with open(
            os.path.join(output_path, f"api_statistics{output_suffix}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Total APIs: {len(api_apis)}\n")
            f.write(f"Total API calls: {total_calls}\n\n")
            for api, count in sorted_api_counts:
                f.write(f"{api}: {count} ({api_percentages[api]:.2f}%)\n")
        print(f"[ConfigSerializer] Write detailed statistics to api_statistics{output_suffix}.txt")

    @staticmethod
    def parse_trace_stacks(input_path: str, output_path: str, output_suffix: str = ""):
        print(f"[ConfigSerializer] Start to parse stacks from: {input_path}")
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"[ConfigSerializer] Trace file not found, skipping: {input_path}")
            return

        if os.path.splitext(input_path)[1].lower() != ".yaml":
            print(f"[ConfigSerializer] Trace file is not a yaml file, skipping: {input_path}")
            return

        api_stack = {}
        completed_apis = set()
        Loader = yaml.CLoader if hasattr(yaml, "CLoader") else yaml.FullLoader

        with open(input_path, encoding="utf-8") as f:
            for call in yaml.load_all(f, Loader=Loader):
                api_name = call.get("api", "UnknownAPI")
                if api_name in completed_apis:
                    continue
                if api_name not in api_stack:
                    api_stack[api_name] = {"3stacks": []}

                stacks_list = api_stack[api_name]["3stacks"]
                if len(stacks_list) < 3:
                    stacks_list.append(call.get("stack", []))
                    if len(stacks_list) == 3:
                        completed_apis.add(api_name)

        print(
            f"[ConfigSerializer] Read {len(api_stack)} unique traces from {os.path.basename(input_path)}"
        )

        output_file = os.path.join(output_path, f"api_stacks{output_suffix}.yaml")
        Dumper = yaml.CDumper if hasattr(yaml, "CDumper") else yaml.Dumper
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                api_stack,
                f,
                Dumper=Dumper,
                allow_unicode=True,
                sort_keys=True,
                default_flow_style=False,
                indent=2,
            )
        print(
            f"[ConfigSerializer] Write {len(api_stack)} api stacks to {os.path.basename(output_file)}"
        )
