from __future__ import annotations

import argparse
import csv
import functools
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = (
    'paddle.matmul(Tensor(paddle.Size([1, 32, 1024]),"bfloat16"), '
    'Tensor(paddle.Size([1024, 32768]),"bfloat16"), )'
)
DEFAULT_STAGE_PREFIXES = (
    "APIConfig.",
    "case.",
    "APITestAccuracy.",
    "APITestPaddleOnly.",
    "APITestBase.",
    "Paddle2TorchConverter.",
    "TensorConfig.",
)
HEAT_CHARS = " .:-=+*#%@"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class StageRecord:
    iteration: int
    phase: str
    stage: str
    duration_ms: float


class StageRecorder:
    def __init__(self) -> None:
        self.records: list[StageRecord] = []
        self._current_iteration = -1
        self._current_phase = "unknown"

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @contextmanager
    def iteration(self, index: int, phase: str):
        previous_iteration = self._current_iteration
        previous_phase = self._current_phase
        self._current_iteration = index
        self._current_phase = phase
        try:
            yield
        finally:
            self._current_iteration = previous_iteration
            self._current_phase = previous_phase

    @contextmanager
    def time_stage(self, stage: str, sync_devices: bool = True):
        if sync_devices:
            synchronize_devices()
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            if sync_devices:
                synchronize_devices()
            end_ns = time.perf_counter_ns()
            self.records.append(
                StageRecord(
                    iteration=self._current_iteration,
                    phase=self._current_phase,
                    stage=stage,
                    duration_ms=(end_ns - start_ns) / 1_000_000,
                )
            )


_PATCHED_METHODS: list[tuple[type[Any], str, Any]] = []


def synchronize_devices() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    try:
        import paddle

        device = paddle.device.get_device()
        if device.startswith("gpu"):
            device_id = 0
            if ":" in device:
                device_id = int(device.rsplit(":", 1)[1])
            paddle.base.core._cuda_synchronize(paddle.CUDAPlace(device_id))
    except Exception:
        pass


def set_device(device: str) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.rsplit(":", 1)[-1])

    try:
        import paddle

        paddle.device.set_device(device)
    except Exception as err:
        print(f"[warn] failed to set paddle device {device}: {err}")

    try:
        import torch

        if device.startswith("gpu"):
            torch.cuda.set_device(int(device.rsplit(":", 1)[1]))
    except Exception as err:
        print(f"[warn] failed to set torch device {device}: {err}")


def patch_public_methods(cls: type[Any], recorder: StageRecorder) -> None:
    for name, value in list(cls.__dict__.items()):
        if name.startswith("_"):
            continue
        if isinstance(value, (staticmethod, classmethod, property)):
            continue
        if not callable(value):
            continue
        if getattr(value, "__paddleapitest_heatmap_wrapped__", False):
            continue

        stage_name = f"{cls.__name__}.{name}"

        @functools.wraps(value)
        def wrapped(self, *args, __method=value, __stage=stage_name, **kwargs):
            with recorder.time_stage(__stage):
                return __method(self, *args, **kwargs)

        wrapped.__paddleapitest_heatmap_wrapped__ = True
        setattr(cls, name, wrapped)
        _PATCHED_METHODS.append((cls, name, value))


def restore_patches() -> None:
    while _PATCHED_METHODS:
        cls, name, value = _PATCHED_METHODS.pop()
        setattr(cls, name, value)


def init_logging(output_dir: Path) -> None:
    try:
        from tester.api_config.logging import init_log
    except Exception as err:
        print(f"[warn] failed to import logging: {err}")
        return

    log_dir = output_dir / "paddleapitest_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        init_log(str(log_dir))
    except Exception as err:
        print(f"[warn] failed to initialize logging: {err}")


def install_stage_hooks(recorder: StageRecorder, mode: str) -> None:
    from tester.api_config.config_analyzer import APIConfig, TensorConfig
    from tester.base import APITestBase

    patch_public_methods(APIConfig, recorder)
    patch_public_methods(TensorConfig, recorder)
    patch_public_methods(APITestBase, recorder)

    if mode == "accuracy":
        from tester.accuracy import APITestAccuracy
        from tester.paddle_to_torch.converter import Paddle2TorchConverter

        patch_public_methods(APITestAccuracy, recorder)
        patch_public_methods(Paddle2TorchConverter, recorder)
    elif mode == "paddle_only":
        from tester.paddle_only import APITestPaddleOnly

        patch_public_methods(APITestPaddleOnly, recorder)
    else:
        msg = f"unsupported mode: {mode}"
        raise ValueError(msg)


def build_case(config: str, mode: str, recorder: StageRecorder, args: argparse.Namespace) -> Any:
    from tester.api_config.config_analyzer import APIConfig

    if mode == "accuracy":
        from tester.accuracy import APITestAccuracy

        case_cls = APITestAccuracy
    elif mode == "paddle_only":
        from tester.paddle_only import APITestPaddleOnly

        case_cls = APITestPaddleOnly
    else:
        msg = f"unsupported mode: {mode}"
        raise ValueError(msg)

    with recorder.time_stage("APIConfig.__init__"):
        api_config = APIConfig(config)
    with recorder.time_stage(f"{case_cls.__name__}.__init__"):
        if mode == "accuracy":
            return case_cls(
                api_config,
                accuracy_compare_mode=args.accuracy_compare_mode,
                accuracy_max_elements=args.accuracy_max_elements,
                accuracy_sample_size=args.accuracy_sample_size,
                use_gpu_mode=args.use_gpu_mode,
            )
        return case_cls(api_config)


def run_one_iteration(
    config: str, mode: str, recorder: StageRecorder, args: argparse.Namespace
) -> None:
    with recorder.time_stage("case.build"):
        case = build_case(config, mode, recorder, args)
    with recorder.time_stage("case.test_total"):
        case.test()


def aggregate_records(records: list[StageRecord], phase: str) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], float] = defaultdict(float)
    phase_records = [record for record in records if record.phase == phase]
    phase_iterations = sorted({record.iteration for record in phase_records})
    iteration_map = {iteration: index for index, iteration in enumerate(phase_iterations)}
    for record in phase_records:
        values[(iteration_map[record.iteration], record.stage)] += record.duration_ms
    return values


def sorted_stages(
    values: dict[tuple[int, str], float], stage_prefixes: tuple[str, ...]
) -> list[str]:
    totals: dict[str, float] = defaultdict(float)
    for (_iteration, stage), duration_ms in values.items():
        if stage_prefixes and not stage.startswith(stage_prefixes):
            continue
        totals[stage] += duration_ms
    return [
        stage for stage, _duration in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def write_records_csv(path: Path, records: list[StageRecord], warmup: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "phase", "stage", "duration_ms", "is_warmup"])
        for record in records:
            writer.writerow(
                [
                    record.iteration,
                    record.phase,
                    record.stage,
                    f"{record.duration_ms:.6f}",
                    int(record.iteration < warmup),
                ]
            )


def write_summary_csv(path: Path, values: dict[tuple[int, str], float], stages: list[str]) -> None:
    total = sum(values.values())
    iterations = sorted({iteration for iteration, _stage in values})
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "total_ms",
                "avg_ms",
                "p50_ms",
                "p90_ms",
                "max_ms",
                "calls_or_iters",
                "share_pct",
            ]
        )
        for stage in stages:
            samples = [values.get((iteration, stage), 0.0) for iteration in iterations]
            nonzero_samples = [sample for sample in samples if sample > 0]
            if not nonzero_samples:
                continue
            stage_total = sum(nonzero_samples)
            sorted_samples = sorted(nonzero_samples)
            p90_index = min(len(sorted_samples) - 1, math.ceil(len(sorted_samples) * 0.9) - 1)
            writer.writerow(
                [
                    stage,
                    f"{stage_total:.6f}",
                    f"{statistics.mean(nonzero_samples):.6f}",
                    f"{statistics.median(nonzero_samples):.6f}",
                    f"{sorted_samples[p90_index]:.6f}",
                    f"{max(nonzero_samples):.6f}",
                    len(nonzero_samples),
                    f"{stage_total / total * 100:.2f}" if total else "0.00",
                ]
            )


def write_cache_events_csv(path: Path) -> None:
    from tester.api_config import config_analyzer

    cached_numpy_events = getattr(config_analyzer, "cached_numpy_events", [])
    cached_gpu_input_events = getattr(config_analyzer, "cached_gpu_input_events", [])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cache_type",
                "dtype",
                "shape",
                "generation_kind",
                "hit",
                "numel",
                "nbytes",
                "duration_ms",
            ]
        )
        for event in cached_numpy_events:
            writer.writerow(
                [
                    "numpy",
                    event.get("dtype", ""),
                    event.get("shape", ""),
                    event.get("generation_kind", ""),
                    int(event.get("hit", False)),
                    event.get("numel", ""),
                    event.get("nbytes", ""),
                    f"{event.get('duration_ms', 0.0):.6f}",
                ]
            )
        for event in cached_gpu_input_events:
            writer.writerow(
                [
                    "gpu_input",
                    event.get("dtype", ""),
                    event.get("shape", ""),
                    "input",
                    int(event.get("hit", False)),
                    event.get("numel", ""),
                    "",
                    f"{event.get('duration_ms', 0.0):.6f}",
                ]
            )


def make_heatmap_text(
    values: dict[tuple[int, str], float],
    stages: list[str],
    repeat: int,
    top_k: int,
    scale: str,
) -> str:
    selected_stages = stages[:top_k]
    if not selected_stages:
        return "no stage records collected"

    max_value = max(values.values(), default=0.0)
    if max_value <= 0:
        max_value = 1.0

    lines = []
    lines.append("PaddleAPITest execution-time heatmap")
    lines.append(
        "char scale: low=' ' -> high='@'; each cell is accumulated stage time per iteration"
    )
    lines.append("")
    header = "stage".ljust(52) + " | " + "".join(f"{idx % 10}" for idx in range(repeat))
    lines.append(header)
    lines.append("-" * len(header))

    for stage in selected_stages:
        chars = []
        row_values = []
        for iteration in range(repeat):
            value = values.get((iteration, stage), 0.0)
            row_values.append(value)
            if scale == "log":
                ratio = math.log1p(value) / math.log1p(max_value)
            else:
                ratio = value / max_value
            char_index = min(len(HEAT_CHARS) - 1, max(0, int(round(ratio * (len(HEAT_CHARS) - 1)))))
            chars.append(HEAT_CHARS[char_index])
        total = sum(row_values)
        avg = statistics.mean(row_values) if row_values else 0.0
        lines.append(
            f"{stage[:52].ljust(52)} | {''.join(chars)}  total={total:.3f}ms avg={avg:.3f}ms"
        )

    return "\n".join(lines)


def maybe_write_png(
    path: Path,
    values: dict[tuple[int, str], float],
    stages: list[str],
    repeat: int,
    top_k: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    selected_stages = stages[:top_k]
    if not selected_stages:
        return False

    matrix = [
        [values.get((iteration, stage), 0.0) for iteration in range(repeat)]
        for stage in selected_stages
    ]
    height = max(4, min(18, len(selected_stages) * 0.35 + 2))
    width = max(8, min(24, repeat * 0.25 + 6))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title("PaddleAPITest stage execution time heatmap")
    ax.set_xlabel("iteration")
    ax.set_ylabel("stage")
    ax.set_yticks(range(len(selected_stages)))
    ax.set_yticklabels(selected_stages, fontsize=7)
    ax.set_xticks(range(repeat))
    ax.set_xticklabels([str(i) for i in range(repeat)], fontsize=7)
    fig.colorbar(image, ax=ax, label="duration ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PaddleAPITest itself for one matmul config and generate execution-time heatmaps."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="PaddleAPITest API config string")
    parser.add_argument(
        "--mode",
        choices=("accuracy", "paddle_only"),
        default="accuracy",
        help="PaddleAPITest test mode to profile",
    )
    parser.add_argument("--device", default="gpu:0", help="Paddle device, e.g. gpu:0")
    parser.add_argument(
        "--warmup", type=int, default=2, help="warmup iterations excluded from summary"
    )
    parser.add_argument("--repeat", type=int, default=10, help="measured iterations after warmup")
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="number of hottest stages rendered in heatmap",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "log"),
        default="log",
        help="ASCII heatmap color scale",
    )
    parser.add_argument(
        "--output-dir",
        default="profiler_outputs/paddleapitest_matmul_heatmap",
        help="directory for csv/txt/png outputs",
    )
    parser.add_argument(
        "--stage-prefix",
        action="append",
        default=None,
        help=(
            "stage prefix to include in summary/heatmap; can be repeated. "
            "Defaults to PaddleAPITest-related classes."
        ),
    )
    parser.add_argument(
        "--use-cached-numpy",
        action="store_true",
        help="set USE_CACHED_NUMPY=True before importing PaddleAPITest modules",
    )
    parser.add_argument(
        "--use-gpu-cache-mode",
        action="store_true",
        help="enable engine-equivalent GPU tensor generation, GPU compare, and allocator reuse",
    )
    parser.add_argument(
        "--use-gpu-input-cache",
        action="store_true",
        help="explicitly generate/cache supported inputs on GPU for aggressive profiling",
    )
    parser.add_argument(
        "--gpu-input-cache-min-numel",
        type=int,
        default=0,
        help="minimum numel for --use-gpu-input-cache",
    )
    parser.add_argument(
        "--gpu-input-cache-apis",
        default="paddle.matmul,paddle.Tensor.matmul,paddle.mm,paddle.bmm",
        help="comma-separated API names eligible for --use-gpu-input-cache",
    )
    parser.add_argument(
        "--accuracy-compare-mode",
        choices=("full", "gpu_full", "sampled", "metadata"),
        default="full",
        help="accuracy compare mode; non-full modes are explicit profiling/screening modes",
    )
    parser.add_argument(
        "--accuracy-max-elements",
        type=int,
        default=1_000_000,
        help="element threshold for sampled/metadata compare modes",
    )
    parser.add_argument(
        "--accuracy-sample-size",
        type=int,
        default=4096,
        help="number of sampled elements for sampled compare mode",
    )
    parser.add_argument(
        "--log-cached-numpy",
        action="store_true",
        help="print cached numpy fill events while profiling",
    )
    parser.add_argument(
        "--cached-numpy-max-entries",
        type=int,
        default=32,
        help="maximum shape-aware cached numpy entries; <=0 disables eviction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_cached_numpy:
        os.environ["USE_CACHED_NUMPY"] = "True"
    if args.use_gpu_mode:
        os.environ["USE_GPU_MODE"] = "True"
        os.environ["SKIP_GPU_CLEANUP"] = "True"
    if args.use_gpu_input_cache:
        os.environ["USE_GPU_INPUT_CACHE"] = "True"
        os.environ["GPU_INPUT_CACHE_MIN_NUMEL"] = str(args.gpu_input_cache_min_numel)
        os.environ["GPU_INPUT_CACHE_APIS"] = args.gpu_input_cache_apis
    if args.use_gpu_input_cache or args.accuracy_compare_mode != "full":
        os.environ["SKIP_GPU_CLEANUP"] = "True"
    if args.log_cached_numpy:
        os.environ["LOG_CACHED_NUMPY"] = "True"
    os.environ["CACHED_NUMPY_MAX_ENTRIES"] = str(args.cached_numpy_max_entries)
    os.environ["RECORD_CACHE_EVENTS"] = "True"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_device(args.device)
    init_logging(output_dir)

    recorder = StageRecorder()
    install_stage_hooks(recorder, args.mode)

    total_iterations = args.warmup + args.repeat
    try:
        for iteration in range(total_iterations):
            label = "warmup" if iteration < args.warmup else "measure"
            print(f"[{label}] iteration {iteration + 1}/{total_iterations}", flush=True)
            with recorder.iteration(iteration, label):
                run_one_iteration(args.config, args.mode, recorder, args)
    finally:
        restore_patches()

    warmup_values = aggregate_records(recorder.records, "warmup")
    measure_values = aggregate_records(recorder.records, "measure")
    stage_prefixes = tuple(args.stage_prefix) if args.stage_prefix else DEFAULT_STAGE_PREFIXES
    warmup_stages = sorted_stages(warmup_values, stage_prefixes)
    measure_stages = sorted_stages(measure_values, stage_prefixes)

    records_csv = output_dir / "matmul_paddleapitest_stage_records.csv"
    warmup_summary_csv = output_dir / "matmul_paddleapitest_warmup_stage_summary.csv"
    summary_csv = output_dir / "matmul_paddleapitest_stage_summary.csv"
    heatmap_txt = output_dir / "matmul_paddleapitest_heatmap.txt"
    heatmap_png = output_dir / "matmul_paddleapitest_heatmap.png"
    cache_events_csv = output_dir / "matmul_paddleapitest_cache_events.csv"

    write_records_csv(records_csv, recorder.records, args.warmup)
    write_summary_csv(warmup_summary_csv, warmup_values, warmup_stages)
    write_summary_csv(summary_csv, measure_values, measure_stages)
    write_cache_events_csv(cache_events_csv)
    heatmap = make_heatmap_text(measure_values, measure_stages, args.repeat, args.top_k, args.scale)
    heatmap_txt.write_text(heatmap + "\n")
    png_written = maybe_write_png(
        heatmap_png, measure_values, measure_stages, args.repeat, args.top_k
    )

    print("\n" + heatmap)
    print("\noutputs:")
    print(f"  records:        {records_csv}")
    print(f"  warmup summary: {warmup_summary_csv}")
    print(f"  summary:        {summary_csv}")
    print(f"  cache events:   {cache_events_csv}")
    print(f"  heatmap:        {heatmap_txt}")
    if png_written:
        print(f"  png:            {heatmap_png}")
    else:
        print("  png:            skipped, matplotlib is unavailable")


if __name__ == "__main__":
    main()
