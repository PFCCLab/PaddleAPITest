from __future__ import annotations

import abc
import functools
import importlib
import inspect
import os
import pkgutil
import traceback
import types
from typing import TYPE_CHECKING, Any

import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING:
    from config_serializer import ConfigSerializer


class TracingHook(abc.ABC):
    """钩子的抽象基类"""

    def __init__(
        self,
        serializer: ConfigSerializer,
        level: int,
        dialect: FrameworkDialect | None = None,
    ):
        self.serializer = serializer
        self.level = level
        self.dialect = dialect

    @abc.abstractmethod
    def install(self):
        pass

    @abc.abstractmethod
    def uninstall(self):
        pass


# SetattrHook can not hook torch C API
class SetattrHook(TracingHook):
    def __init__(
        self,
        serializer: ConfigSerializer,
        level: int,
        dialect: FrameworkDialect,
    ):
        super().__init__(serializer, level, dialect)
        self._original_apis: dict[str, Any] = {}
        self._module_cache: dict[str, Any] = {}

    def _get_api_parent_and_name(self, api_name: str) -> tuple[Any | None, str | None]:
        parts = api_name.split(".")
        if len(parts) < 2:
            return None, None

        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            try:
                if module_name in self._module_cache:
                    module = self._module_cache[module_name]
                else:
                    module = importlib.import_module(module_name)
                    self._module_cache[module_name] = module

                parent_obj = module
                for attr in parts[i:-1]:
                    parent_obj = getattr(parent_obj, attr)
                return parent_obj, parts[-1]
            except (ImportError, AttributeError):
                continue
        return None, None

    @staticmethod
    def _create_wrapper(
        api_name: str,
        original_api: Any,
        serializer: ConfigSerializer,
        level: int,
    ):
        @torch.compiler.disable  # disable this in low version of torch
        @functools.wraps(original_api)
        def wrapper(*args, **kwargs):
            output = original_api(*args, **kwargs)
            serializer.dump_call(api_name, args, kwargs, level=level)
            return output

        return wrapper

    def install(self):
        if self.dialect is None:
            return
        api_list = self.dialect.discover_apis() + self.dialect.discover_custom_ops()

        # api_path = os.path.join(os.path.dirname(__file__), "trace_output/api_list.yaml")
        # with open(api_path, "w") as f:
        #     yaml.dump(api_list, f)

        print(f"[SetattrHook] Attempting to patch {len(api_list)} APIs...")
        patched_count, skipped_count = 0, 0
        for api_name in api_list:
            if api_name in self._original_apis:
                # print(f"[SetattrHook] Skipping {api_name}: Already patched")
                continue

            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if not parent_obj or not func_name or not hasattr(parent_obj, func_name):
                print(f"[SetattrHook] Skipping {api_name}: Could not resolve path.")
                continue

            try:
                original_api = getattr(parent_obj, func_name)
                wrapper = None

                if isinstance(
                    original_api,
                    (
                        types.FunctionType,
                        types.BuiltinFunctionType,
                        types.MethodType,
                        types.BuiltinMethodType,
                    ),
                ):
                    wrapper = self._create_wrapper(
                        api_name, original_api, self.serializer, self.level
                    )
                elif isinstance(original_api, (classmethod, staticmethod)):
                    original_func = original_api.__func__
                    wrapped_func = self._create_wrapper(
                        api_name, original_func, self.serializer, self.level
                    )
                    wrapper = type(original_api)(wrapped_func)
                elif isinstance(original_api, property) and original_api.fget and original_api.fset:
                    wrapped_getter = self._create_wrapper(
                        f"{api_name}.fget",
                        original_api.fget,
                        self.serializer,
                        self.level,
                    )
                    wrapper = property(
                        wrapped_getter,
                        original_api.fset,
                        original_api.fdel,
                        original_api.__doc__,
                    )

                if wrapper:
                    setattr(parent_obj, func_name, wrapper)
                    self._original_apis[api_name] = original_api
                    patched_count += 1
                else:
                    print(f"[SetattrHook] Could not patch {api_name}")
                    skipped_count += 1
            except (TypeError, AttributeError) as e:
                error_msg = str(e).lower()
                if (
                    "immutable type" in error_msg
                    or "can't set attribute" in error_msg
                    or "read-only" in error_msg
                ):
                    print(f"[SetattrHook] Skip non-writable API '{api_name}'.")
                    skipped_count += 1
                else:
                    print(f"[SetattrHook] Could not patch {api_name}: {e}")
            except Exception as e:
                print(f"[SetattrHook] Could not patch {api_name}: {e}")

        print(
            f"[SetattrHook] Patched {patched_count} APIs. Skipped {skipped_count} non-writable APIs."
        )

        # api_path = os.path.join(os.path.dirname(__file__), "trace_output/api_list_wrap.yaml")
        # with open(api_path, "w") as f:
        #     yaml.dump(list(self._original_apis.keys()), f)

    def uninstall(self):
        print(f"[SetattrHook] Restoring {len(self._original_apis)} patched APIs...")
        for api_name, original_api in self._original_apis.items():
            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if parent_obj and func_name:
                try:
                    setattr(parent_obj, func_name, original_api)
                except Exception as e:
                    print(f"[SetattrHook] Error restoring API '{api_name}': {e}")
        self._original_apis.clear()
        self._module_cache.clear()
        print("[SetattrHook] Restoration complete.")


class TorchFunctionModeTracer(torch.overrides.TorchFunctionMode):
    def __init__(self, serializer: ConfigSerializer, level: int, dialect: FrameworkDialect):
        self.serializer = serializer
        self.level = level
        self.disable_torch_api_list = getattr(dialect, "disable_torch_api_list", False)
        self.target_apis = getattr(dialect, "target_apis", [])

        # skip these for duplicate property access of paddle.Tensor in SetattrHook
        # (SetattrHook and TorchFunctionHook are installed at the same time)
        self.ignored_apis = {
            "torch.Tensor.shape.__get__",
            "torch.Tensor.dtype.__get__",
            "torch.Tensor.device.__get__",
        }

        # disable this may result in recursion error, but this will ignore factory APIs (e.g. paddle.randn)
        self.ignored_functions = torch.overrides.get_ignored_functions()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in self.ignored_functions:
            return func(*args, **kwargs)

        output = func(*args, **kwargs)

        api_name = torch.overrides.resolve_name(func)
        if not api_name:
            if hasattr(func, "__module__"):
                api_name = f"{func.__module__}.{func.__name__}"
            elif hasattr(func, "__objclass__"):
                api_name = (
                    f"{func.__objclass__.__module__}.{func.__objclass__.__name__}.{func.__name__}"
                )
            else:
                api_name = f"unknown.{func.__name__}"
                print(f"Unknown func: {func}, type: {type(func)}")

        if api_name not in self.ignored_apis:
            self.serializer.dump_call(api_name, args, kwargs, level=self.level)
        return output


class TorchFunctionHook(TracingHook):
    def __init__(self, serializer: ConfigSerializer, level: int, dialect: FrameworkDialect):
        super().__init__(serializer, level, dialect)
        self.tracing_mode = TorchFunctionModeTracer(serializer, level, dialect)

    def install(self):
        print("[TorchFunctionHook] Enabling __torch_function__ tracing mode...")
        self.tracing_mode.__enter__()
        print("[TorchFunctionHook] Mode enabled.")

    def uninstall(self):
        print("[TorchFunctionHook] Disabling __torch_function__ tracing mode...")
        self.tracing_mode.__exit__(None, None, None)
        print("[TorchFunctionHook] Mode disabled.")


class TorchDispatchModeTracer(TorchDispatchMode):
    def __init__(self, serializer: ConfigSerializer, level: int, dialect: FrameworkDialect):
        self.serializer = serializer
        self.level = level
        self.disable_torch_api_list = getattr(dialect, "disable_torch_api_list", False)
        self.target_apis = getattr(dialect, "target_apis", [])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        output = func(*args, **kwargs)

        api_name = func.name()
        self.serializer.dump_call(api_name, args, kwargs, level=self.level)
        return output


class TorchDispatchHook(TracingHook):
    def __init__(self, serializer: ConfigSerializer, level: int, dialect: FrameworkDialect):
        super().__init__(serializer, level, dialect)
        self.tracing_mode = TorchDispatchModeTracer(serializer, level, dialect)

    def install(self):
        print("[TorchDispatchHook] Enabling __torch_dispatch__ tracing mode...")
        self.tracing_mode.__enter__()
        print("[TorchDispatchHook] Mode enabled.")

    def uninstall(self):
        print("[TorchDispatchHook] Disabling __torch_dispatch__ tracing mode...")
        self.tracing_mode.__exit__(None, None, None)
        print("[TorchDispatchHook] Mode disabled.")


class FrameworkDialect(abc.ABC):
    """框架方言的抽象基类"""

    @abc.abstractmethod
    def get_framework_name(self) -> str:
        """返回框架名称"""
        raise NotImplementedError

    def discover_apis(self) -> list[str]:
        """返回框架API列表"""
        return []

    def discover_custom_ops(self) -> list[str]:
        """返回自定义算子API列表"""
        return []

    @abc.abstractmethod
    def serialize_special_type(self, item: Any) -> dict | None:
        """序列化框架所特有的数据类型"""
        raise NotImplementedError

    @abc.abstractmethod
    def format_special_type(self, item: Any) -> str:
        """格式化框架所特有的数据类型"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_hooks(
        self, serializer: ConfigSerializer, level: int | list[int], **kwargs
    ) -> list[TracingHook]:
        """获取跟踪钩子, 用于在API调用时进行记录"""
        raise NotImplementedError

    @classmethod
    def get_dialect(cls, framework_name: str) -> FrameworkDialect:
        dialect_map = {"torch": PyTorchDialect}
        dialect_class = dialect_map.get(framework_name)
        if not dialect_class:
            raise ValueError(f"Unsupported framework: {framework_name}")
        return dialect_class()


class PyTorchDialect(FrameworkDialect):
    """PyTorch方言实现"""

    # Only takes effect in SetattrHook
    MODULE_BLACKLIST = {
        "torch._dynamo.config",
        "torch._dynamo.test_case",
        "torch._dynamo.test_minifier_common",
        "torch._functorch.config",
        "torch._inductor.config",
        "torch._inductor.test_case",
        "torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity",
        "torch.backends._coreml.preprocess",
        "torch.compiler.config",
        "torch.contrib._tensorboard_vis",
        "torch.distributed._tools.sac_ilp",
        "torch.distributed.elastic.rendezvous"  # multi modules
        "torch.fx.experimental._config",
        "torch.onnx._internal.exporter",  # multi modules
        "torch.onnx._internal.fx",  # multi modules
        "torch.testing._internal",  # multi modules
        "torch.utils._cxx_pytree",
        "torch.utils.model_dump",
        "torch.utils.serialization.config",
        "torch.utils.tensorboard",
        # modules above here will cause some errors
        "torch._C",
        "torch._decomp",
        "torch._dynamo",
        "torch._functorch",
        "torch._inductor",
        "torch._jit_internal",
        "torch._ops",
        "torch._tensor",
        "torch._tensor_str",
        "torch.overrides",
        "torch.utils._python_dispatch",
        "torch.distributed._shard.checkpoint",
        "torch.distributed._shard.sharding_spec",
        "torch.distributed._sharding_spec",
        # modules below here are optional
        # "torch._export",
        # "torch._guards",
        # "torch._guards",
        # "torch._higher_order_ops",
        # "torch._jit_internal",
        # "torch._library",
        # "torch._linalg_utils",
        # "torch._lobpcg",
        # "torch._lowrank",
        # "torch._meta_registrations",
        # "torch._prims",
        # "torch._refs",
        # "torch._subclasses",
        # "torch._vmap_internals",
        # "torch.compiler._cache",
        # "torch.distributed._shard",
        # "torch.distributed._tools",
        # "torch.export._trace",
        # "torch.fx._graph_pickler",
        # "torch.jit._decomposition_utils",
        # "torch.jit._decompositions",
        # "torch.jit._recursive",
        # "torch.jit._script",
        # "torch.jit._trace",
        # "torch.masked._ops",
        # "torch.nested._internal",
        # "torch.utils._foreach_utils",
    }

    # Only takes effect in SetattrHook
    IGNORE_ATTRIBUTES = {
        # should skip
        "__bytes__",
        "__class__",
        "__delattr__",
        "__delete__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__get__",
        "__getattr__",
        "__getattribute__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__module__",
        "__new__",
        "__prepare__",
        "__set__",
        "__setattr__",
        "__sizeof__",
        "__slots__",
        "__weakref__",
        "xreplace",
        # recommended to skip
        "__call__",
        "__format__",
        "__iter__",
        "__repr__",
        "__str__",
        "__instancecheck__",
        "__subclasscheck__",
        "__subclasshook__",
        "__getstate__",
        "__setstate__",
        "__enter__",
        "__exit__",
    }

    # Only takes effect in SetattrHook
    IGNORE_CLASSES_OR_METHODS = {
        # classes
        "torch._utils.CallbackRegistry",
        "torch.cuda._gpu_trace.CallbackRegistry",
        "torch.cuda._sanitizer.StreamSynchronizations",
        "torch.cuda._sanitizer._TensorsAccessed",
        "torch.xpu._gpu_trace.CallbackRegistry",
        # methods
        "torch.Event.from_ipc_handle",
        "torch.Size.__class_getitem__",
        "torch.autograd.function._is_setup_context_defined",
        "torch.distributed.reduce_op",
        "torch.fx.Node._pretty_print_target",
        "torch.fx.experimental.unification.multipledispatch.dispatcher.str_signature",
        "torch.mtia.Event.from_ipc_handle",
        "torch.nn.functional.handle_torch_function",
        "torch.nn.functional.has_torch_function_unary",
        "torch.optim.Optimizer.profile_hook_step",  # it will be overridden by subclass of Optimizer
        "torch.nn.parallel.DistributedDataParallel._get_active_ddp_module",  # it will stuck RWKV
        "torch.nn.Module.get_extra_state",  # it will cause RuntimeError, and be overridden by subclass of Module
        "torch.nn.Module.set_extra_state",  # it will cause RuntimeError, and be overridden by subclass of Module
        "torch.fx.wrap",  # it will cause NotImplementedError
    }

    def get_framework_name(self) -> str:
        return "torch"

    # Only takes effect in SetattrHook
    def discover_apis(self) -> list[str]:
        """使用pkgutil遍历torch包"""
        print(f"[{self.__class__.__name__}] Discovering APIs for '{self.get_framework_name()}'...")

        base_module = importlib.import_module(self.get_framework_name())
        api_set: set[str] = set()
        modules = {base_module}

        if hasattr(base_module, "__path__"):
            for module_info in pkgutil.walk_packages(
                base_module.__path__, prefix=base_module.__name__ + "."
            ):
                if any(module_info.name.startswith(pattern) for pattern in self.MODULE_BLACKLIST):
                    # print(
                    #     f"[Discovery Info] Skipping blacklisted module: {module_info.name}"
                    # )
                    continue
                try:
                    sub_module = importlib.import_module(module_info.name)
                    modules.add(sub_module)
                except Exception as e:
                    print(
                        f"[{self.__class__.__name__}] Could not import module {module_info.name}: {e}"
                    )
                    continue

        print(f"[{self.__class__.__name__}] Discovered {len(modules)} modules to inspect.")

        for module in modules:
            for member_name in dir(module):
                full_name = f"{module.__name__}.{member_name}"
                try:
                    obj = getattr(module, member_name)
                    if (
                        not hasattr(obj, "__module__")
                        or not obj.__module__
                        or not obj.__module__.startswith("torch")
                    ):
                        continue
                    if not self.disable_torch_api_list and full_name not in self.target_apis:
                        continue
                    if full_name in self.IGNORE_CLASSES_OR_METHODS:
                        continue
                    if isinstance(
                        obj,
                        (
                            types.FunctionType,
                            types.BuiltinFunctionType,
                            types.MethodType,
                            types.BuiltinMethodType,
                            staticmethod,
                            classmethod,
                        ),
                    ):
                        api_set.add(full_name)
                    elif inspect.isclass(obj):
                        # custom op class should be skip
                        if issubclass(obj, torch.autograd.Function):
                            continue
                        for cls_member_name, cls_member in inspect.getmembers(obj):
                            if cls_member_name in self.IGNORE_ATTRIBUTES:
                                continue
                            full_cls_name = f"{full_name}.{cls_member_name}"
                            if full_cls_name in self.IGNORE_CLASSES_OR_METHODS:
                                continue
                            # it will be overridden by subclass of Optimizer
                            if cls_member_name in {
                                "profile_hook_step",
                                "get_extra_state",
                                "set_extra_state",
                            }:
                                continue
                            if isinstance(
                                cls_member,
                                (
                                    types.FunctionType,
                                    types.BuiltinFunctionType,
                                    types.MethodType,
                                    types.BuiltinMethodType,
                                    staticmethod,
                                    classmethod,
                                ),
                            ) or (
                                isinstance(cls_member, property)
                                and cls_member.fget
                                and cls_member.fset
                            ):
                                api_set.add(full_cls_name)
                except Exception as e:
                    traceback.print_exc()
                    print(
                        f"[Discovery Warning] Could not inspect {module.__name__}.{member_name}: {e}"
                    )
                    continue

        api_list = sorted(api_set)
        print(f"[{self.__class__.__name__}] Discovered {len(api_list)} native APIs.")
        return api_list

    # Only takes effect in SetattrHook
    def discover_custom_ops(self) -> list[str]:
        # TODO(@cangtianhuang): implemente me
        return []

    _special_type_handlers = {
        torch.Tensor: lambda item: {
            "type": "torch.Tensor",
            "shape": list(item.shape),
            "dtype": str(item.dtype),
            "device": str(item.device),
        },
        torch.dtype: lambda item: {"type": "torch.dtype", "value": str(item)},
        torch.device: lambda item: {"type": "torch.device", "value": str(item)},
        torch.memory_format: lambda item: {
            "type": "torch.memory_format",
            "value": str(item),
        },
        torch.layout: lambda item: {"type": "torch.layout", "value": str(item)},
        torch.Size: lambda item: {  # Added handler for torch.Size
            "type": "torch.Size",
            "value": list(item),
        },
        # TODO(@cangtianhuang): add more serialization logic here
    }

    def serialize_special_type(self, item: Any) -> dict | None:
        handler = self._special_type_handlers.get(type(item))
        return handler(item) if handler else None

    _format_handlers = {
        "torch.Tensor": lambda item: f'Tensor({item["shape"]}, "{item["dtype"].replace("torch.", "")}")',
        "torch.dtype": lambda item: f'"{item["value"].replace("torch.", "")}"',
        "torch.device": lambda item: f'"{item["value"]}"',
        "torch.memory_format": lambda item: f'"{item["value"]}"',
        "torch.layout": lambda item: f'"{item["value"].replace("torch.", "")}"',
        "torch.Size": lambda item: f"list{item['value']}",
        # TODO(@cangtianhuang): add more formatting logic here
    }

    def format_special_type(self, item: dict) -> str | None:
        handler = self._format_handlers.get(item.get("type", ""))
        return handler(item) if handler else None

    def get_hooks(self, serializer, levels: list[int], **kwargs) -> list[TracingHook]:
        self.target_apis = []
        self.disable_torch_api_list = kwargs.get("disable_torch_api_list", False)
        if not self.disable_torch_api_list and 0 in levels:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "api_list",
                "torch_api_list.yaml",
            )
            with open(yaml_path, encoding="utf-8") as f:
                self.target_apis = yaml.safe_load(f)
            print(f"[{self.__class__.__name__}] Loaded {len(self.target_apis)} target APIs.")

        hooks = []
        hook_map = {0: SetattrHook, 1: TorchFunctionHook, 2: TorchDispatchHook}
        for level in levels:
            hook_class = hook_map.get(level)
            if hook_class:
                hooks.append(hook_class(serializer, level, self))
            else:
                raise ValueError(f"Invalid level: {level}")
        return hooks
