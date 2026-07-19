from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_DUMP_DIR = "tester/api_config/test_log/dump_case"


def parse_strict_bool(value: bool | str) -> bool:
    """Parse supported boolean spellings, rejecting every other value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"expected one of true/1/yes/y or false/0/no/n, got {value!r}")


def resolve_dump_options(
    cli_use_dump: bool | None,
    cli_dump_dir: str | None,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> tuple[bool, str]:
    """Resolve each dump field independently with CLI > env > default precedence."""
    env = os.environ if environ is None else environ
    if cli_use_dump is not None:
        use_dump = cli_use_dump
    elif "USE_DUMP" in env:
        try:
            use_dump = parse_strict_bool(env["USE_DUMP"])
        except ValueError as err:
            raise ValueError(f"invalid USE_DUMP: {err}") from err
    else:
        use_dump = False

    if cli_dump_dir is not None:
        dump_dir = cli_dump_dir or DEFAULT_DUMP_DIR
    else:
        dump_dir = env.get("DUMP_DIR") or DEFAULT_DUMP_DIR
    return use_dump, dump_dir


class TeeStream:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        self.flush()
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self):
        return self.primary.fileno()


class DumpContext:
    def __init__(
        self,
        dump_dir: str | os.PathLike[str],
        api_config: str | None = None,
        *,
        auto_start: bool = True,
    ):
        self.root = Path(dump_dir)
        self.tensors_dir = self.root / "tensors"
        self.metadata_path = self.root / "dump.yaml"
        self.api_config = api_config or ""
        self.current_phase = "created"
        self._data: dict[str, Any] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensors_dir.mkdir(parents=True, exist_ok=True)
        self._data = self._init_metadata(auto_start=auto_start)
        self.current_phase = self._data.get("current_phase", "created")
        if auto_start:
            self._data["environment"] = collect_environment(self._data["run"])
            self._append_event("engine_start")
            self._flush_metadata()

    def _init_metadata(self, *, auto_start: bool) -> dict[str, Any]:
        if not auto_start and self.metadata_path.exists():
            try:
                with self.metadata_path.open(encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
        return {
            "schema_version": 1,
            "created_at": _now_text(),
            "created_timestamp": time.time(),
            "api_config": self.api_config,
            "current_phase": self.current_phase,
            "status": None,
            "run": _collect_run_info(),
            "environment": {},
            "events": [],
            "tensors": [],
        }

    def _flush_metadata(self) -> None:
        self._data["current_phase"] = self.current_phase
        tmp = self.metadata_path.with_suffix(".yaml.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                _make_yaml_safe(self._data),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        tmp.replace(self.metadata_path)

    def _append_event(self, name: str, **data: Any) -> None:
        self.current_phase = data.pop("phase", name)
        payload = {
            "event": name,
            "time": _now_text(),
            "timestamp": time.time(),
            "pid": os.getpid(),
            "current_phase": self.current_phase,
        }
        payload.update(data)
        self._data.setdefault("events", []).append(payload)

    def event(self, name: str, **data: Any) -> None:
        self._append_event(name, **data)
        self._flush_metadata()

    def error_event(self, name: str, err: BaseException) -> None:
        self.event(
            name,
            error_type=type(err).__name__,
            error=str(err),
            traceback="".join(traceback.format_exception(type(err), err, err.__traceback__)),
        )

    def finalize(self, status: str, **data: Any) -> None:
        payload = {
            "name": status,
            "time": _now_text(),
            "timestamp": time.time(),
            "pid": os.getpid(),
            "current_phase": self.current_phase,
        }
        payload.update(data)
        self._data["status"] = payload
        self._append_event(f"final_{status}", **data)
        self._flush_metadata()

    def save_tensors(self, stem: str, obj: Any, framework: str | None = None) -> None:
        arrays: dict[str, np.ndarray] = {}
        items: list[dict[str, Any]] = []
        _collect_tensor_items(obj, stem, arrays, items, framework=framework)
        path = self.tensors_dir / f"{stem}.npz"
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
        stat = path.stat()
        entry = {
            "name": stem,
            "framework": framework,
            "file": str(path.relative_to(self.root)),
            "format": "npz",
            "size_bytes": stat.st_size,
            "saved_at": _now_text(),
            "items": _make_yaml_safe(items),
        }
        tensors = self._data.setdefault("tensors", [])
        tensors[:] = [item for item in tensors if item.get("name") != stem]
        tensors.append(entry)
        self._append_event(
            f"save_{stem}_done",
            tensor_file=entry["file"],
            size_bytes=stat.st_size,
        )
        self._flush_metadata()

    @contextlib.contextmanager
    def tee_output(self):
        log_path = self.root / "case.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = TeeStream(old_stdout, f)
            sys.stderr = TeeStream(old_stderr, f)
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr


def collect_environment(process: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "captured_at": _now_text(),
        "process": process or _collect_run_info(),
        "system": _collect_system_info(),
        "env": _collect_relevant_env(),
        "gpu": _collect_gpu_info(),
        "cuda_runtime": _collect_cuda_info(),
        "framework_runtime": _collect_framework_runtime(),
        "python_packages": _collect_python_packages(),
        "pip_freeze": _run_command([sys.executable, "-m", "pip", "freeze"], timeout=30).get(
            "stdout_lines", []
        ),
    }


def _collect_run_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }


def _collect_system_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_build": platform.python_build(),
        "python_compiler": platform.python_compiler(),
        "libc": platform.libc_ver(),
        "uname": platform.uname()._asdict(),
    }
    os_release = Path("/etc/os-release")
    if os_release.exists():
        info["os_release"] = os_release.read_text(encoding="utf-8", errors="replace")
    return info


def _collect_relevant_env() -> dict[str, str]:
    prefixes = (
        "CUDA",
        "CUDNN",
        "CUBLAS",
        "CUPTI",
        "NCCL",
        "NVIDIA",
        "NV",
        "FLAGS_",
        "PADDLE",
        "TORCH",
        "PYTORCH",
        "PYTHON",
        "OMP",
        "MKL",
        "OPENBLAS",
        "BLAS",
        "LAPACK",
        "LD_",
    )
    names = {
        "PATH",
        "PYTHONPATH",
        "LIBRARY_PATH",
        "CPATH",
        "USE_GPU_MODE",
        "USE_CACHED_NUMPY",
        "USE_DUMP",
        "DUMP_DIR",
        "SKIP_GPU_CLEANUP",
    }
    redacted = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "AK", "SK")
    env = {}
    for key, value in sorted(os.environ.items()):
        if key in names or key.startswith(prefixes):
            if any(part in key.upper() for part in redacted):
                env[key] = "<redacted>"
            else:
                env[key] = value
    return env


def _collect_gpu_info() -> dict[str, Any]:
    queries = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "vbios_version",
        "compute_cap",
        "memory.total",
        "memory.free",
        "memory.used",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.limit",
        "power.draw",
        "temperature.gpu",
    ]
    result = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(queries)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    info: dict[str, Any] = {"nvidia_smi_query": result}
    if result.get("returncode") == 0:
        gpus = []
        for line in result.get("stdout_lines", []):
            values = [part.strip() for part in line.split(",")]
            gpus.append(dict(zip(queries, values)))
        info["gpus"] = gpus
    info["nvidia_smi"] = _run_command(["nvidia-smi"], timeout=15)
    return info


def _collect_cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "nvcc": _run_command(["nvcc", "--version"], timeout=15),
        "ldconfig_cuda_libraries": _collect_cuda_libraries(),
    }
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        info["cuda_home"] = cuda_home
        info["cuda_version_file"] = _read_optional(
            Path(cuda_home) / "version.json"
        ) or _read_optional(Path(cuda_home) / "version.txt")
    return info


def _collect_cuda_libraries() -> list[str] | dict[str, str]:
    if not shutil.which("ldconfig"):
        return {"error": "ldconfig not found"}
    result = _run_command(["ldconfig", "-p"], timeout=15)
    if result.get("returncode") != 0:
        return result
    patterns = (
        "cuda",
        "cudart",
        "cublas",
        "cudnn",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "cutensor",
        "nccl",
        "nvrtc",
    )
    return [
        line
        for line in result.get("stdout_lines", [])
        if any(name in line.lower() for name in patterns)
    ]


def _collect_framework_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {}

    def collect_paddle():
        import paddle

        version = getattr(paddle, "version", None)
        return {
            "version": getattr(version, "full_version", getattr(paddle, "__version__", None)),
            "cuda": version.cuda() if version and hasattr(version, "cuda") else None,
            "cudnn": version.cudnn() if version and hasattr(version, "cudnn") else None,
            "nccl": version.nccl() if version and hasattr(version, "nccl") else None,
            "mkl": version.mkl() if version and hasattr(version, "mkl") else None,
            "compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
        }

    def collect_torch():
        import torch

        return {
            "version": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None,
            "nccl": torch.cuda.nccl.version()
            if torch.cuda.is_available()
            and hasattr(torch.cuda, "nccl")
            and hasattr(torch.cuda.nccl, "version")
            else None,
            "hip": torch.version.hip,
            "cuda_available": torch.cuda.is_available(),
        }

    info["paddle"] = _safe_call(collect_paddle)
    info["torch"] = _safe_call(collect_torch)
    info["numpy"] = {"version": np.__version__}
    return info


def _collect_python_packages() -> dict[str, str]:
    packages = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _safe_call(fn):
    try:
        return fn()
    except Exception as err:
        return {"error": str(err)}


def _run_command(command: list[str], timeout: int = 10) -> dict[str, Any]:
    if not command or not shutil.which(command[0]):
        return {"command": command, "error": f"{command[0] if command else ''} not found"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_lines": completed.stdout.splitlines(),
            "stderr_lines": completed.stderr.splitlines(),
        }
    except Exception as err:
        return {"command": command, "error": str(err)}


def _read_optional(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return None


def _collect_tensor_items(
    obj: Any,
    path: str,
    arrays: dict[str, np.ndarray],
    items: list[dict[str, Any]],
    *,
    framework: str | None,
) -> None:
    if obj is None:
        items.append({"path": path, "kind": "none"})
        return
    if isinstance(obj, dict):
        items.append({"path": path, "kind": "dict", "keys": list(obj.keys())})
        for key, value in obj.items():
            _collect_tensor_items(value, f"{path}/{key}", arrays, items, framework=framework)
        return
    if isinstance(obj, (list, tuple)):
        items.append({"path": path, "kind": type(obj).__name__, "length": len(obj)})
        for index, value in enumerate(obj):
            _collect_tensor_items(value, f"{path}/{index}", arrays, items, framework=framework)
        return

    meta: dict[str, Any] = {"path": path}
    array = tensor_to_numpy(obj, meta)
    if array is None:
        items.append(_make_yaml_safe(meta))
        return
    key = _unique_key(_safe_key(path), arrays)
    arrays[key] = array
    meta.update(
        {
            "key": key,
            "saved_dtype": str(arrays[key].dtype),
            "saved_shape": list(arrays[key].shape),
            "nbytes": int(arrays[key].nbytes),
        }
    )
    if framework and "framework" not in meta:
        meta["framework"] = framework
    items.append(_make_yaml_safe(meta))


def tensor_to_numpy(obj: Any, meta: dict[str, Any]) -> np.ndarray | None:
    module = type(obj).__module__
    type_name = type(obj).__name__
    if module.startswith("paddle") and type_name == "Tensor":
        meta.update(
            {
                "kind": "tensor",
                "framework": "paddle",
                "type": "Tensor",
                "dtype": str(getattr(obj, "dtype", "")),
                "shape": list(getattr(obj, "shape", [])),
                "place": str(getattr(obj, "place", "")),
                "stop_gradient": getattr(obj, "stop_gradient", None),
            }
        )
        return obj.numpy()
    if module.startswith("torch") and type_name == "Tensor":
        meta.update(
            {
                "kind": "tensor",
                "framework": "torch",
                "type": "Tensor",
                "dtype": str(getattr(obj, "dtype", "")),
                "shape": list(getattr(obj, "shape", [])),
                "device": str(getattr(obj, "device", "")),
                "requires_grad": getattr(obj, "requires_grad", None),
            }
        )
        tensor = obj.detach().cpu().contiguous()
        try:
            return tensor.numpy()
        except Exception:
            meta["stored_as"] = "raw_uint8"
            return np.frombuffer(tensor.numpy(force=True).tobytes(), dtype=np.uint8)
    if isinstance(obj, np.ndarray):
        meta.update({"kind": "ndarray", "dtype": str(obj.dtype), "shape": list(obj.shape)})
        return obj
    if isinstance(obj, (bool, int, float, str, np.generic)):
        array = np.asarray(obj)
        meta.update(
            {
                "kind": type(obj).__name__,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "value": obj.item() if isinstance(obj, np.generic) else obj,
            }
        )
        return array
    meta.update({"kind": "repr", "type": f"{module}.{type_name}", "repr": repr(obj)})
    return None


def _safe_key(path: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", path).strip("_")
    return key or "value"


def _unique_key(key: str, arrays: dict[str, np.ndarray]) -> str:
    if key not in arrays:
        return key
    index = 1
    while f"{key}_{index}" in arrays:
        index += 1
    return f"{key}_{index}"


def _make_yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_yaml_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def dump_enabled() -> bool:
    return parse_strict_bool(os.environ.get("USE_DUMP", "false"))


def finalize_from_parent(status: str, **data: Any) -> None:
    if not dump_enabled():
        return
    ctx = DumpContext(
        os.environ.get("DUMP_DIR") or DEFAULT_DUMP_DIR,
        auto_start=False,
    )
    if not ctx._data.get("environment"):
        ctx._data["environment"] = collect_environment()
    ctx._append_event(status, **data)
    ctx._data["status"] = {
        "name": status,
        "time": _now_text(),
        "timestamp": time.time(),
        "pid": os.getpid(),
        "current_phase": ctx.current_phase,
        **data,
    }
    ctx._append_event(f"final_{status}", **data)
    ctx._flush_metadata()
