from __future__ import annotations

import argparse
import copy
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("test_pipeline/run_config.yaml")
DEFAULT_RETEST_ERROR_CONFIGS = [
    "api_config_accuracy_error.txt",
    "api_config_paddle_error.txt",
    "api_config_timeout.txt",
    "api_config_crash.txt",
    "api_config_oom.txt",
    "api_config_cuda_error.txt",
    "api_config_match_error.txt",
    "api_config_numpy_error.txt",
    "api_config_torch_error.txt",
    "api_config_paddle_to_torch_failed.txt",
]

TOP_LEVEL_KEYS = {"name", "runner", "env", "input", "output", "retest", "engine_args"}
RUNNER_KEYS = {"engine", "foreground", "dry_run", "pid_file"}
INPUT_KEYS = {"api_config", "api_config_file", "api_config_file_pattern"}
OUTPUT_KEYS = {"log_dir"}
RETEST_KEYS = {
    "enabled",
    "rounds",
    "error_configs",
    "log_dir_template",
    "skip_unavailable",
    # Backward-compatible legacy keys. Keep accepted, but do not show in templates.
    "skip_missing",
    "skip_empty",
}
ENGINE_ARG_TYPES = {
    "paddle_only": bool,
    "paddle_cinn": bool,
    "accuracy": bool,
    "paddle_gpu_performance": bool,
    "torch_gpu_performance": bool,
    "paddle_torch_gpu_performance": bool,
    "accuracy_stable": bool,
    "paddle_custom_device": bool,
    "test_amp": bool,
    "num_gpus": int,
    "num_workers_per_gpu": int,
    "gpu_ids": str,
    "required_memory": (int, float),
    "test_cpu": bool,
    "use_cached_numpy": bool,
    "use_gpu_cache_mode": bool,
    "atol": (int, float),
    "rtol": (int, float),
    "manual_threshold_config_file": str,
    "test_tol": bool,
    "test_backward": bool,
    "timeout": int,
    "show_runtime_status": bool,
    "random_seed": int,
    "custom_device_vs_gpu": bool,
    "custom_device_vs_gpu_mode": str,
    "bitwise_alignment": bool,
    "generate_failed_tests": bool,
    "exit_on_error": bool,
    "use_compute_sanitizer": bool,
    "sanitizer_command": str,
    "sanitizer_error_exitcode": int,
    "_sanitizer_child": bool,
}


def reject_unknown_keys(section: str, mapping: dict[str, Any], allowed_keys: set[str]) -> None:
    unknown = sorted(set(mapping) - allowed_keys)
    if unknown:
        allowed = ", ".join(sorted(allowed_keys))
        raise ValueError(f"{section} 包含未知字段: {', '.join(unknown)}；允许字段: {allowed}")


def require_type(path: str, value: Any, expected_type: type | tuple[type, ...]) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if expected_type is int or expected_type in {(int, float), (float, int)}:
            actual = "int" if expected_type is int else "number"
            raise TypeError(f"{path} 类型应为 {actual}，当前为 bool")
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_name = " | ".join(t.__name__ for t in expected_type)
        else:
            expected_name = expected_type.__name__
        raise TypeError(f"{path} 类型应为 {expected_name}，当前为 {type(value).__name__}")


def validate_yaml_config(config: dict[str, Any]) -> None:
    reject_unknown_keys("root", config, TOP_LEVEL_KEYS)
    if "name" in config:
        require_type("name", config["name"], str)

    runner = ensure_mapping(config, "runner")
    reject_unknown_keys("runner", runner, RUNNER_KEYS)
    if "engine" in runner and runner["engine"] not in {"engineV2", "engineV4"}:
        raise ValueError("runner.engine 仅支持 engineV2 或 engineV4")
    for key in ("foreground", "dry_run"):
        if key in runner:
            require_type(f"runner.{key}", runner[key], bool)
    if "pid_file" in runner:
        require_type("runner.pid_file", runner["pid_file"], str)

    env = ensure_mapping(config, "env")
    for key, value in env.items():
        if not isinstance(key, str):
            raise TypeError("env 的 key 必须是字符串")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"env.{key} 类型应为 scalar")

    input_config = ensure_mapping(config, "input")
    reject_unknown_keys("input", input_config, INPUT_KEYS)
    configured_inputs = [value for value in input_config.values() if value]
    if len(configured_inputs) != 1:
        raise ValueError(
            "input.api_config、input.api_config_file、input.api_config_file_pattern 必须且只能配置一个"
        )
    for key in INPUT_KEYS:
        if key in input_config:
            require_type(f"input.{key}", input_config[key], str)

    output = ensure_mapping(config, "output")
    reject_unknown_keys("output", output, OUTPUT_KEYS)
    if not output.get("log_dir"):
        raise ValueError("output.log_dir 不能为空")
    require_type("output.log_dir", output["log_dir"], str)

    retest = retest_config(config)
    reject_unknown_keys("retest", retest, RETEST_KEYS)
    if "enabled" in retest:
        require_type("retest.enabled", retest["enabled"], bool)
    if "rounds" in retest:
        require_type("retest.rounds", retest["rounds"], int)
        if retest["rounds"] < 1:
            raise ValueError("retest.rounds 必须 >= 1")
    if "error_configs" in retest and retest["error_configs"] is not None:
        if not isinstance(retest["error_configs"], list) or not all(
            isinstance(item, str) for item in retest["error_configs"]
        ):
            raise TypeError("retest.error_configs 必须是字符串列表")
    if "log_dir_template" in retest:
        require_type("retest.log_dir_template", retest["log_dir_template"], str)
    for key in ("skip_unavailable", "skip_missing", "skip_empty"):
        if key in retest:
            require_type(f"retest.{key}", retest[key], bool)

    engine_args = ensure_mapping(config, "engine_args")
    reject_unknown_keys("engine_args", engine_args, set(ENGINE_ARG_TYPES))
    for key, value in engine_args.items():
        require_type(f"engine_args.{key}", value, ENGINE_ARG_TYPES[key])
    if engine_args.get("custom_device_vs_gpu_mode") not in {None, "upload", "download"}:
        raise ValueError("engine_args.custom_device_vs_gpu_mode 仅支持 upload 或 download")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是 YAML mapping: {path}")
    return config


def parse_key_value(value: str, option_name: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option_name} 需要 KEY=VALUE 格式: {value}")
    key, parsed_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"{option_name} 的 KEY 不能为空: {value}")
    return key, parsed_value


def parse_engine_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def bool_to_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
    if value is None:
        value = {}
        config[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是 mapping")
    return value


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def default_pid_file(config_path: Path, config: dict[str, Any]) -> Path:
    name = str(config.get("name") or config_path.stem)
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return PROJECT_ROOT / ".run" / f"{safe_name}.pid"


def set_input_config(input_config: dict[str, Any], key: str, value: str | None) -> None:
    if value:
        for k in INPUT_KEYS:
            input_config[k] = None
        input_config[key] = value


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    runner = ensure_mapping(config, "runner")
    input_config = ensure_mapping(config, "input")
    output = ensure_mapping(config, "output")
    engine_args = ensure_mapping(config, "engine_args")
    env = ensure_mapping(config, "env")

    if args.engine:
        runner["engine"] = args.engine
    if args.foreground:
        runner["foreground"] = True
    if args.background:
        runner["foreground"] = False
    if args.dry_run:
        runner["dry_run"] = True

    set_input_config(input_config, "api_config", args.api_config)
    set_input_config(input_config, "api_config_file", args.api_config_file)
    set_input_config(input_config, "api_config_file_pattern", args.api_config_file_pattern)
    if args.log_dir:
        output["log_dir"] = args.log_dir

    simple_engine_overrides = {
        "timeout": args.timeout,
        "num_gpus": args.num_gpus,
        "num_workers_per_gpu": args.num_workers_per_gpu,
        "gpu_ids": args.gpu_ids,
        "required_memory": args.required_memory,
    }
    for key, value in simple_engine_overrides.items():
        if value is not None:
            engine_args[key] = value

    for item in args.set_env:
        key, value = parse_key_value(item, "--set-env")
        env[key] = value

    for item in args.engine_arg:
        key, value = parse_key_value(item, "--engine-arg")
        engine_args[key.replace("-", "_")] = parse_engine_value(value)


def validate_config(config_path: Path, config: dict[str, Any]) -> tuple[str, Path, Path]:
    if not (PROJECT_ROOT / "tester").is_dir():
        raise RuntimeError(f"请在 PaddleAPITest 项目根目录附近执行，未找到 tester/: {PROJECT_ROOT}")

    runner = ensure_mapping(config, "runner")
    input_config = ensure_mapping(config, "input")
    output = ensure_mapping(config, "output")

    engine = runner.get("engine") or "engineV4"
    if engine not in {"engineV2", "engineV4"}:
        raise ValueError(f"runner.engine 仅支持 engineV2 或 engineV4: {engine}")
    engine_path = PROJECT_ROOT / f"{engine}.py"
    if not engine_path.exists():
        raise FileNotFoundError(f"engine 文件不存在: {engine_path}")

    api_config = input_config.get("api_config")
    api_config_file = input_config.get("api_config_file")
    api_config_pattern = input_config.get("api_config_file_pattern")
    configured_inputs = [
        value for value in (api_config, api_config_file, api_config_pattern) if value
    ]
    if len(configured_inputs) != 1:
        raise ValueError(
            "input.api_config、input.api_config_file、input.api_config_file_pattern 必须且只能配置一个"
        )

    log_dir = output.get("log_dir")
    if not log_dir:
        raise ValueError("output.log_dir 不能为空")

    pid_file_value = runner.get("pid_file")
    pid_file = (
        resolve_project_path(pid_file_value)
        if pid_file_value
        else default_pid_file(config_path, config)
    )
    return str(engine), engine_path, pid_file


def build_engine_command(engine: str, config: dict[str, Any], passthrough: list[str]) -> list[str]:
    input_config = ensure_mapping(config, "input")
    output = ensure_mapping(config, "output")
    engine_args = ensure_mapping(config, "engine_args")

    command = [sys.executable, f"{engine}.py"]
    api_config = input_config.get("api_config")
    api_config_file = input_config.get("api_config_file")
    api_config_pattern = input_config.get("api_config_file_pattern")
    if api_config:
        command.append(f"--api_config={api_config}")
    if api_config_file:
        command.append(f"--api_config_file={api_config_file}")
    if api_config_pattern:
        command.append(f"--api_config_file_pattern={api_config_pattern}")
    command.append(f"--log_dir={output['log_dir']}")

    for key, value in engine_args.items():
        if value is None:
            continue
        command.append(f"--{key}={bool_to_cli(value)}")

    command.extend(passthrough)
    return command


def command_to_display(command: list[str]) -> str:
    return shlex.join(command)


def input_value_from_config(config: dict[str, Any]) -> str | None:
    input_config = ensure_mapping(config, "input")
    return (
        input_config.get("api_config")
        or input_config.get("api_config_file")
        or input_config.get("api_config_file_pattern")
    )


def set_api_config_file(config: dict[str, Any], api_config_file: str) -> None:
    input_config = ensure_mapping(config, "input")
    input_config["api_config"] = None
    input_config["api_config_file"] = api_config_file
    input_config["api_config_file_pattern"] = None


def retest_error_name(error_config: str) -> str:
    name = Path(error_config).name
    if name.startswith("api_config_"):
        name = name[len("api_config_") :]
    if name.endswith(".txt"):
        name = name[: -len(".txt")]
    return name


def format_retest_log_dir(
    template: str,
    *,
    base_log_dir: str,
    previous_log_dir: str,
    round_index: int,
    error_config: str,
) -> str:
    error_name = retest_error_name(error_config)
    return template.format(
        base_log_dir=base_log_dir.rstrip("/"),
        previous_log_dir=previous_log_dir.rstrip("/"),
        round=round_index,
        error_config=Path(error_config).name,
        error_name=error_name,
    )


def retest_config(config: dict[str, Any]) -> dict[str, Any]:
    retest = config.get("retest") or {}
    if not isinstance(retest, dict):
        raise ValueError("配置项 retest 必须是 mapping")
    return retest


def is_retest_enabled(config: dict[str, Any]) -> bool:
    return bool(retest_config(config).get("enabled"))


def retest_error_configs(config: dict[str, Any]) -> list[str]:
    retest = retest_config(config)
    error_configs = retest.get("error_configs") or DEFAULT_RETEST_ERROR_CONFIGS
    if not isinstance(error_configs, list) or not all(
        isinstance(item, str) for item in error_configs
    ):
        raise ValueError("retest.error_configs 必须是字符串列表")
    return error_configs


def should_skip_retest_input(path: Path, *, skip_unavailable: bool) -> bool:
    if not path.exists():
        if skip_unavailable:
            print(f"[RETEST] 跳过不存在的失败配置: {display_path(path)}")
            return True
        raise FileNotFoundError(f"失败配置文件不存在: {path}")
    if path.stat().st_size == 0:
        if skip_unavailable:
            print(f"[RETEST] 跳过空失败配置: {display_path(path)}")
            return True
        raise ValueError(f"失败配置文件为空: {path}")
    return False


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process(pid_file: Path) -> int:
    pid = read_pid(pid_file)
    if pid is None:
        print("未找到 PID 文件，没有正在运行的任务")
        return 0
    if process_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        print(f"已终止进程组 PGID={pid}")
    else:
        print(f"进程 PID={pid} 已不存在")
    pid_file.unlink(missing_ok=True)
    return 0


def latest_log(log_dir: Path) -> Path | None:
    try:
        logs = sorted(
            log_dir.glob("log_*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return logs[0] if logs else None
    except (OSError, FileNotFoundError):
        return None


def show_status(pid_file: Path, engine: str, log_dir: Path) -> int:
    pid = read_pid(pid_file)
    if pid is None:
        print("无运行记录（PID 文件不存在）")
        return 0
    if not process_running(pid):
        print(f"\033[31m已结束\033[0m  PID={pid} (进程不存在)")
        pid_file.unlink(missing_ok=True)
        return 0

    print(f"\033[32m运行中\033[0m  PID={pid}  引擎={engine}")
    try:
        children_output = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
        children = sum(1 for line in children_output.splitlines() if line.strip())
    except subprocess.CalledProcessError:
        children = 0
    print(f"  Worker 进程数: {children}")

    try:
        elapsed = subprocess.check_output(["ps", "-o", "etime=", "-p", str(pid)], text=True).strip()
    except subprocess.CalledProcessError:
        elapsed = "unknown"
    print(f"  已运行: {elapsed}")

    log = latest_log(log_dir)
    if log:
        print(f"  最新日志: {display_path(log)}")
    return 0


def prepare_log_file(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"log_{timestamp}.log"


def cleanup_pid_when_done(pid: int, pid_file: Path) -> None:
    while process_running(pid):
        time.sleep(5)
    recorded_pid = read_pid(pid_file)
    if recorded_pid == pid:
        pid_file.unlink(missing_ok=True)


def run_foreground(command: list[str], env: dict[str, str], log_file: Path) -> int:
    print("\n\033[36m[前台模式] Ctrl+C 终止\033[0m")
    print(f"日志同时写入: {display_path(log_file)}")
    print("")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with log_file.open("a", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
    return process.wait()


def run_background(
    command: list[str], env: dict[str, str], log_file: Path, pid_file: Path, engine: str
) -> int:
    if pid_file.exists():
        old_pid = read_pid(pid_file)
        if old_pid and process_running(old_pid):
            print(f"\033[33m警告: 已有运行中的任务 PID={old_pid}\033[0m")
            print(f"使用 --stop 终止后再启动，或删除 {display_path(pid_file)} 强制启动")
            return 1
        pid_file.unlink(missing_ok=True)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    cleaner_code = (
        "import os,sys,time\npid=int(sys.argv[1])\npid_file=sys.argv[2]\n"
        "while True:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except ProcessLookupError:\n"
        "        break\n"
        "    except PermissionError:\n"
        "        pass\n"
        "    time.sleep(5)\n"
        "try:\n"
        "    recorded=open(pid_file).read().strip()\n"
        "except FileNotFoundError:\n"
        "    recorded=''\n"
        "if recorded == str(pid):\n"
        "    try:\n"
        "        os.remove(pid_file)\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
    )
    cleaner = subprocess.Popen(
        [sys.executable, "-c", cleaner_code, str(process.pid), str(pid_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    cleaner.wait()

    time.sleep(1)
    if not process_running(process.pid):
        print(f"\033[31m错误: {engine} 启动失败\033[0m")
        print(f"查看日志: tail -50 {display_path(log_file)}")
        pid_file.unlink(missing_ok=True)
        return 1

    print("")
    print(f"\033[32m启动成功! PID={process.pid}\033[0m")
    print("")
    print("常用操作:")
    print("  查看状态:  python run.py -c <config> --status")
    print("  终止任务:  python run.py -c <config> --stop")
    print(f"  跟踪日志:  tail -f {display_path(log_file)}")
    print("  GPU监控:   watch -n 1 nvidia-smi")
    print("")
    print("进程已在后台运行，关闭终端不影响执行")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleAPITest runner")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="YAML 任务配置文件")
    parser.add_argument("--stop", action="store_true", help="终止后台任务")
    parser.add_argument("--status", action="store_true", help="查看后台任务状态")
    parser.add_argument("--dry-run", action="store_true", help="只打印最终命令，不执行")
    parser.add_argument("--foreground", action="store_true", help="前台运行")
    parser.add_argument("--background", action="store_true", help="后台运行")
    parser.add_argument("--engine", choices=["engineV2", "engineV4"], help="覆盖 runner.engine")
    parser.add_argument("--api-config", help="覆盖 input.api_config")
    parser.add_argument("--api-config-file", help="覆盖 input.api_config_file")
    parser.add_argument("--api-config-file-pattern", help="覆盖 input.api_config_file_pattern")
    parser.add_argument("--log-dir", help="覆盖 output.log_dir")
    parser.add_argument("--timeout", type=int, help="覆盖 engine_args.timeout")
    parser.add_argument("--num-gpus", type=int, help="覆盖 engine_args.num_gpus")
    parser.add_argument(
        "--num-workers-per-gpu", type=int, help="覆盖 engine_args.num_workers_per_gpu"
    )
    parser.add_argument("--gpu-ids", help="覆盖 engine_args.gpu_ids")
    parser.add_argument("--required-memory", type=float, help="覆盖 engine_args.required_memory")
    parser.add_argument(
        "--set-env", action="append", default=[], help="追加或覆盖环境变量 KEY=VALUE"
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        help="追加或覆盖 engine 参数 KEY=VALUE",
    )
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="-- 后原样透传给 engine")
    return parser.parse_args()


def run_once(
    *,
    config_path: Path,
    config: dict[str, Any],
    passthrough: list[str],
    force_foreground: bool = False,
    label: str | None = None,
) -> int:
    engine, _engine_path, pid_file = validate_config(config_path, config)
    output = ensure_mapping(config, "output")
    runner = ensure_mapping(config, "runner")
    env_config = ensure_mapping(config, "env")
    log_dir = resolve_project_path(output["log_dir"])
    command = build_engine_command(engine, config, passthrough)

    env = os.environ.copy()
    for key, value in env_config.items():
        if value is not None:
            env[str(key)] = str(value)

    print("── PaddleAPITest ──────────────────────────")
    if label:
        print(f"  轮次:    {label}")
    print(f"  配置:    {display_path(config_path)}")
    print(f"  引擎:    {engine}.py")
    print(f"  输入:    {input_value_from_config(config)}")
    print(f"  日志:    {output['log_dir']}")
    engine_args = ensure_mapping(config, "engine_args")
    print(
        "  GPU:     "
        f"ids={engine_args.get('gpu_ids', '')}  "
        f"workers/gpu={engine_args.get('num_workers_per_gpu', '')}"
    )
    if engine_args.get("timeout") is not None:
        print(f"  超时:    {engine_args['timeout']}s")
    print(f"  命令:    {command_to_display(command)}")
    print("────────────────────────────────────────────")

    dry_run = bool(runner.get("dry_run"))
    if dry_run:
        print("")
        print("[DRY-RUN] 最终命令:")
        print(f"  {command_to_display(command)}")
        if env_config:
            print("[DRY-RUN] 环境变量:")
            for key in sorted(env_config):
                print(f"  {key}={env_config[key]}")
        return 0

    log_file = prepare_log_file(log_dir)
    foreground = force_foreground or bool(runner.get("foreground"))
    if foreground:
        return run_foreground(command, env, log_file)
    return run_background(command, env, log_file, pid_file, engine)


def run_retest_plan(config_path: Path, config: dict[str, Any], passthrough: list[str]) -> int:
    retest = retest_config(config)
    rounds = int(retest.get("rounds", 1))
    if rounds < 1:
        raise ValueError("retest.rounds 必须 >= 1")

    base_output = ensure_mapping(config, "output")
    base_log_dir = str(base_output["log_dir"])
    log_dir_template = retest.get("log_dir_template") or "{base_log_dir}_r{round}_{error_name}"
    skip_unavailable = bool(
        retest.get(
            "skip_unavailable",
            retest.get("skip_missing", retest.get("skip_empty", True)),
        )
    )

    result = run_once(
        config_path=config_path,
        config=config,
        passthrough=passthrough,
        force_foreground=rounds > 1,
        label="round 1/base",
    )
    if result != 0 or rounds == 1:
        return result

    dry_run = bool(ensure_mapping(config, "runner").get("dry_run"))
    previous_log_dir_path = Path(base_log_dir)
    for error_config in retest_error_configs(config):
        previous_log_dir = base_log_dir
        for round_index in range(2, rounds + 1):
            input_path = resolve_project_path(previous_log_dir_path / error_config)
            next_log_dir = format_retest_log_dir(
                log_dir_template,
                base_log_dir=base_log_dir,
                previous_log_dir=previous_log_dir,
                round_index=round_index,
                error_config=error_config,
            )
            if not dry_run and should_skip_retest_input(
                input_path,
                skip_unavailable=skip_unavailable,
            ):
                break

            round_config = copy.deepcopy(config)
            set_api_config_file(round_config, display_path(input_path))
            ensure_mapping(round_config, "output")["log_dir"] = next_log_dir
            result = run_once(
                config_path=config_path,
                config=round_config,
                passthrough=passthrough,
                force_foreground=True,
                label=f"round {round_index}/{retest_error_name(error_config)}",
            )
            if result != 0:
                return result
            previous_log_dir = next_log_dir
            previous_log_dir_path = Path(next_log_dir)
    return 0


def parse_passthrough(args: argparse.Namespace) -> list[str]:
    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return passthrough


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    apply_overrides(config, args)
    validate_yaml_config(config)
    engine, _engine_path, pid_file = validate_config(config_path, config)

    output = ensure_mapping(config, "output")
    log_dir = resolve_project_path(output["log_dir"])

    if args.stop:
        return stop_process(pid_file)
    if args.status:
        return show_status(pid_file, engine, log_dir)

    passthrough = parse_passthrough(args)
    if is_retest_enabled(config):
        return run_retest_plan(config_path, config, passthrough)
    return run_once(config_path=config_path, config=config, passthrough=passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
