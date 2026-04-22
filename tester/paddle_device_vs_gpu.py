from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import paddle
import yaml

from .api_config.config_analyzer import TensorConfig
from .api_config.log_writer import write_to_log
from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU
from .special_compare import SkipComparison, get_backward_compare, get_forward_compare


class _PaddleSkipError(Exception):
    """Raised by _run_paddle when need_skip(paddle_only=True) is True."""


_DEVICE_VS_GPU_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "device_vs_gpu_config.yaml"
)
_device_vs_gpu_atol_rtol: dict = {}
_device_vs_gpu_dtype_atol_rtol: dict = {}

if os.path.exists(_DEVICE_VS_GPU_CONFIG_PATH):
    with open(_DEVICE_VS_GPU_CONFIG_PATH, encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f) or {}
    _device_vs_gpu_atol_rtol = _cfg.get("atol_rtol", {})
    _device_vs_gpu_dtype_atol_rtol = _cfg.get("dtype_atol_rtol", {})
    del _cfg, _f


class APITestPaddleDeviceVSGPU(APITestCustomDeviceVSCPU):
    def __init__(self, api_config, **kwargs):
        # 继承 CustomDevice vs CPU 的基本功能
        super().__init__(api_config, **kwargs)

        # 新增参数
        self.operation_mode = kwargs.get("operation_mode", None)
        self.bos_path = kwargs.get("bos_path", "")
        self.random_seed = kwargs.get("random_seed", 0)
        self.atol = kwargs.get("atol", 1e-2)
        self.rtol = kwargs.get("rtol", 1e-2)
        self.bcecmd_path = Path(kwargs.get("bcecmd_path", "./bcecmd")).resolve()
        self.bos_conf_path = kwargs.get("bos_conf_path", "./conf")

        # HTTP 模式参数
        self.http_host = kwargs.get("http_host", "")
        self.http_port = kwargs.get("http_port", 8089)
        self.http_timeout = kwargs.get("http_timeout", 300)

        # 设置随机种子确保一致性
        if self.random_seed != 0:
            np.random.seed(self.random_seed)
            paddle.seed(self.random_seed)

    def _get_config_hash(self):
        """生成API配置的哈希值，用于文件名"""
        config_str = json.dumps(
            {
                "api_name": self.api_config.api_name,
                "args": [str(arg) for arg in self.api_config.args],
                "kwargs": {k: str(v) for k, v in self.api_config.kwargs.items()},
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_local_device_type(self):
        """获取当前设备的类型，优先复用 engineV2 的检测逻辑。"""
        from engineV2 import detect_device_type

        return detect_device_type()

    def _has_float8_dtype(self):
        """Return True if any arg/kwarg tensor uses a float8 dtype."""

        def _check(cfg):
            if isinstance(cfg, TensorConfig):
                return cfg.dtype in self._FLOAT8_DTYPES
            if isinstance(cfg, (list, tuple)):
                return any(_check(c) for c in cfg)
            if isinstance(cfg, slice):
                return False
            return cfg in self._FLOAT8_DTYPES

        return any(_check(c) for c in self.api_config.args) or any(
            _check(c) for c in self.api_config.kwargs.values()
        )

    def need_skip(self, paddle_only=False):
        # Device vs GPU compares Paddle on XPU against Paddle on GPU — no Torch
        # involved. All conditions in base.need_skip() are Torch-specific (sparse,
        # prod multi-axis, torch_error_skip, float8 dtype), so we do NOT call
        # super() here.
        # XPU cannot create float8 tensors via the float32->cast path.
        if self._get_local_device_type() == "xpu" and self._has_float8_dtype():
            return True
        # XPU does not support sparse kernels.
        # Also covers sparse-related Tensor methods that don't carry "sparse" in their name.
        _SPARSE_APIS = {"paddle.Tensor.coalesce", "paddle.Tensor.is_coalesced"}
        if self._get_local_device_type() == "xpu" and (
            "sparse" in self.api_config.api_name
            or self.api_config.api_name in _SPARSE_APIS
        ):
            return True
        return False

    def _get_filename(self):
        """生成PDTensor文件名（不再包含设备前缀，只依赖随机种子和配置哈希）"""
        return f"{self.random_seed}-{self._get_config_hash()}.pdtensor"

    def _save_tensor_locally(self, output, grads=None):
        """保存结果到本地PDTensor文件"""
        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        filename = self._get_filename()
        local_path = Path(temp_dir) / filename

        # 使用paddle.save保存张量数据
        save_data = {"output": output}
        if grads is not None:
            save_data["grads"] = grads

        paddle.save(save_data, str(local_path))
        print(f"[upload] Saved pdtensor file: {local_path}", flush=True)
        return local_path

    def _build_bos_path(self, filename: str) -> str:
        cleaned = self.bos_path.strip().lstrip("/").rstrip("/")
        return f"bos:/{cleaned}/{filename}"

    def _bcecmd_cp(self, src: str, dst: str, action: str):
        """使用指定的 bcecmd 命令执行 cp 操作"""
        cmd = [
            str(self.bcecmd_path),
            "--conf-path",
            self.bos_conf_path,
            "bos",
            "cp",
            src,
            dst,
        ]
        print(f"[{action}] Running command: {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    def _upload_to_bos(self, local_path):
        """使用 bcecmd 上传文件到 BOS"""
        if not self.bos_path:
            print(f"[upload] No bos_path specified, skip upload", flush=True)
            return

        remote_path = self._build_bos_path(local_path.name)
        try:
            result = self._bcecmd_cp(str(local_path), remote_path, "upload")
            if result.returncode == 0:
                print(f"[upload] Upload succeeded: {remote_path}", flush=True)
                local_path.unlink(missing_ok=True)
            else:
                print(
                    f"[upload] Upload failed: {remote_path}, stderr: {result.stderr}",
                    flush=True,
                )
        except Exception as e:
            print(f"[upload] Upload failed: {e}", flush=True)

    def _download_from_bos(self, filename):
        """使用 bcecmd 从 BOS 下载文件"""
        if not self.bos_path:
            print(f"[download] No bos_path specified, skip download", flush=True)
            return None

        temp_dir = tempfile.gettempdir()
        local_path = Path(temp_dir) / filename

        if local_path.exists():
            print(f"[download] File already exists locally: {local_path}", flush=True)
            return local_path

        remote_path = self._build_bos_path(filename)
        try:
            result = self._bcecmd_cp(remote_path, str(local_path), "download")
            if result.returncode == 0:
                print(f"[download] Download succeeded: {local_path}", flush=True)
                return local_path
            else:
                print(
                    f"[download] Download failed: {remote_path}, stderr: {result.stderr}",
                    flush=True,
                )
                return None
        except Exception as e:
            print(f"[download] Download failed: {e}", flush=True)
            return None

    _FLOAT8_DTYPES = frozenset(["float8_e4m3fn", "float8_e5m2"])

    def _fill_float8_paddle_inputs(self):
        """Create float8 paddle tensors for args that get_paddle_tensor left as None.

        config_analyzer.get_paddle_tensor() returns None for float8 dtypes
        because the generic torch_vs_paddle mode doesn't support them.  In
        device_vs_gpu mode paddle does support float8, so we patch up the
        None entries here without touching any shared code.
        """

        def make_tensor(cfg: TensorConfig) -> paddle.Tensor:
            # Generate float32 numpy data (numpy has no float8 dtype),
            # create a paddle float32 tensor, then cast to float8.
            numpy_data = (np.random.random(cfg.shape) - 0.5).astype("float32")
            t = paddle.to_tensor(numpy_data, dtype="float32", place=cfg.place)
            t = paddle.cast(t, dtype=cfg.dtype)
            t.stop_gradient = False
            return t

        def fix_list(args: list, cfgs: list):
            for i, (arg, cfg) in enumerate(zip(args, cfgs)):
                if (
                    arg is None
                    and isinstance(cfg, TensorConfig)
                    and cfg.dtype in self._FLOAT8_DTYPES
                ):
                    args[i] = make_tensor(cfg)
                elif isinstance(arg, list) and isinstance(cfg, list):
                    fix_list(arg, cfg)

        fix_list(self.paddle_args, self.paddle_args_config)

        for key, val in list(self.paddle_kwargs.items()):
            cfg = self.paddle_kwargs_config.get(key)
            if val is None and isinstance(cfg, TensorConfig) and cfg.dtype in self._FLOAT8_DTYPES:
                self.paddle_kwargs[key] = make_tensor(cfg)

    def _run_paddle(self, device_type: str):
        """在指定设备上运行 Paddle（统一 GPU / XPU / 自定义设备逻辑）。"""
        # Called directly by http_server (bypasses test()), so check paddle-only
        # skips here (e.g. sparse APIs). float8 is intentionally NOT skipped
        # because paddle supports it; need_skip(paddle_only=True) excludes the
        # torch-incompatible float8 check.
        if self.need_skip(paddle_only=True):
            print(f"[skip] {self.api_config.config}", flush=True)
            raise _PaddleSkipError(f"API not supported on this device: {self.api_config.config}")

        try:
            paddle_device_type = device_type
            if device_type == "gpu":
                # engineV2.py sets CUDA_VISIBLE_DEVICES, so paddle will use the correct GPU.
                paddle.set_device("gpu")
            elif device_type == "xpu":
                paddle.set_device(f"xpu:{self.xpu_device_id}")
            elif device_type == self.custom_device_type and self.check_custom_device_available():
                paddle.set_device(f"{self.custom_device_type}:{self.custom_device_id}")
            elif device_type == "cpu":
                paddle.set_device("cpu")
            else:
                print(f"[error] No custom device available", flush=True)
                return None, None

            if not self.ana_paddle_api_info():
                print("ana_paddle_api_info failed", flush=True)
                return None, None

            if not self.gen_numpy_input():
                print("gen_numpy_input failed", flush=True)
                return None, None

            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return None, None

            # device_vs_gpu supports float8 natively; gen_paddle_input() leaves
            # None for float8 tensors (shared guard in config_analyzer). Fix them
            # up here so that only this mode is affected.
            self._fill_float8_paddle_inputs()

            paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)

            # 原地操作返回 None，用被修改的 tensor 替代，使后续对比逻辑不变
            if paddle_output is None:
                api_name = self.api_config.api_name
                if api_name == "paddle.Tensor.__setitem__":
                    paddle_output = self.paddle_args[0]
                elif api_name == "paddle.nn.utils.vector_to_parameters":
                    paddle_output = self.paddle_args[1]

            paddle_grads = None
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                    paddle_output
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_grads = paddle.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                    # sparse=True ops (e.g. embedding) produce SelectedRows gradients.
                    # SelectedRows is NOT detected by is_sparse() / is_dense() — both
                    # return False. Calling to_dense() on SelectedRows triggers a
                    # Segfault in Paddle's C++ layer. Detection: a Tensor that is
                    # neither dense nor any sparse variant is a SelectedRows.
                    # Conversion: numpy() works correctly on SelectedRows; rebuild as
                    # a plain dense Tensor so paddle.save() can serialize it.
                    if paddle_grads is not None:
                        paddle_grads = [
                            paddle.to_tensor(g.numpy())
                            if g is not None
                            and isinstance(g, paddle.Tensor)
                            and not g.is_dense()
                            and not g.is_sparse()
                            and not g.is_sparse_coo()
                            and not g.is_sparse_csr()
                            else g
                            for g in paddle_grads
                        ]

            return paddle_output, paddle_grads

        except Exception as e:
            print(
                f"[paddle {paddle_device_type} error] {self.api_config.config}: {e}",
                flush=True,
            )
            write_to_log("paddle_error", self.api_config.config)
            self._last_error = str(e)
            return None, None

    def _resolve_atol_rtol(self, dtype_str: str) -> tuple[float, float]:
        """按三级优先级解析容差：API+dtype > API default > 全局 dtype > 命令行值。"""
        api_name = self.api_config.api_name
        if api_name in _device_vs_gpu_atol_rtol:
            api_cfg = _device_vs_gpu_atol_rtol[api_name]
            if dtype_str in api_cfg:
                return tuple(float(v) for v in api_cfg[dtype_str])
            if "default" in api_cfg:
                return tuple(float(v) for v in api_cfg["default"])
        if dtype_str in _device_vs_gpu_dtype_atol_rtol:
            return tuple(float(v) for v in _device_vs_gpu_dtype_atol_rtol[dtype_str])
        return self.atol, self.rtol

    def _print_diff(self, label, local_np, remote_np):
        """打印两个 numpy 数组之间的实际最大绝对误差和最大相对误差，返回 (max_abs, max_rel)"""
        a = local_np.astype(np.float64)
        b = remote_np.astype(np.float64)
        abs_diff = np.abs(a - b)
        if abs_diff.size == 0:
            print(
                f"[compare] {label} max_abs_diff=0 (empty tensor), max_rel_diff=0 (empty tensor)",
                flush=True,
            )
            return 0.0, 0.0
        max_abs = float(np.nanmax(abs_diff))
        denom = np.abs(b)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_diff = np.where(denom == 0, abs_diff, abs_diff / denom)
        max_rel = float(np.nanmax(rel_diff))
        print(
            f"[compare] {label} max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}", flush=True
        )
        return max_abs, max_rel

    def _assert_close(self, local_np, remote_np, atol, rtol):
        """手动实现 allclose 语义，避免 np.testing.assert_allclose 在 float16/bfloat16 下
        因内部格式化引发 'Unknown format code g for object of type str' 的 bug。
        判定条件：|actual - desired| <= atol + rtol * |desired|（element-wise，NaN==NaN 视为相等）
        """
        a = local_np.astype(np.float64)
        b = remote_np.astype(np.float64)
        nan_equal = np.isnan(a) == np.isnan(b)
        abs_diff = np.abs(a - b)
        tol = atol + rtol * np.abs(b)
        mismatch = ~nan_equal | ((~np.isnan(a)) & (abs_diff > tol))
        if np.any(mismatch):
            max_abs = float(np.nanmax(abs_diff))
            raise AssertionError(
                f"Arrays not close: max_abs_diff={max_abs:.6g}, atol={atol:.6g}, rtol={rtol:.6g}"
            )

    def _compare_with_downloaded(self, local_output, local_grads, downloaded_tensor):
        """与下载的结果进行对比"""
        try:
            print(f"[compare] Comparing results for {self.api_config.config}", flush=True)

            # 加载下载的数据
            remote_data = paddle.load(str(downloaded_tensor))
            remote_output = remote_data["output"]

            # 对比Forward输出（直接使用Paddle对比）
            try:
                forward_fn = get_forward_compare(self.api_config.api_name)
                if forward_fn is not None:
                    forward_fn(local_output, remote_output, self.api_config, tester=self)
                elif isinstance(local_output, paddle.Tensor) and isinstance(
                    remote_output, paddle.Tensor
                ):
                    # 使用Paddle的对比方法
                    dtype_str = str(local_output.dtype).split(".")[-1]
                    atol, rtol = self._resolve_atol_rtol(dtype_str)
                    local_np = local_output.numpy()
                    remote_np = remote_output.numpy()
                    self._print_diff("Forward", local_np, remote_np)
                    self._assert_close(local_np, remote_np, atol, rtol)
                elif isinstance(local_output, (list, tuple)) and isinstance(
                    remote_output, (list, tuple)
                ):
                    # 列表或元组对比
                    for i, (local_item, remote_item) in enumerate(zip(local_output, remote_output)):
                        if isinstance(local_item, paddle.Tensor) and isinstance(
                            remote_item, paddle.Tensor
                        ):
                            dtype_str = str(local_item.dtype).split(".")[-1]
                            atol, rtol = self._resolve_atol_rtol(dtype_str)
                            local_np = local_item.numpy()
                            remote_np = remote_item.numpy()
                            self._print_diff(f"Forward output[{i}]", local_np, remote_np)
                            self._assert_close(local_np, remote_np, atol, rtol)
                            print(
                                f"[compare] Forward output[{i}] comparison passed",
                                flush=True,
                            )
                else:
                    # 其他情况，尝试转换为numpy对比
                    local_np = (
                        local_output.numpy()
                        if isinstance(local_output, paddle.Tensor)
                        else np.array(local_output)
                    )
                    remote_np = (
                        remote_output.numpy()
                        if isinstance(remote_output, paddle.Tensor)
                        else np.array(remote_output)
                    )
                    np.testing.assert_allclose(
                        local_np,
                        remote_np,
                        atol=self.atol,
                        rtol=self.rtol,
                        equal_nan=True,
                    )

                print(
                    f"[compare] Forward accuracy check passed for {self.api_config.config}",
                    flush=True,
                )
            except SkipComparison as e:
                print(f"[compare] Forward skipped for {self.api_config.config}: {e}", flush=True)
                write_to_log("skip", self.api_config.config)
                return True
            except Exception as e:
                print(
                    f"[compare] Forward accuracy check failed for {self.api_config.config}, error: {e}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False

            # 对比Backward梯度（如果存在且Forward通过）
            if local_grads is not None and "grads" in remote_data:
                remote_grads = remote_data["grads"]

                try:
                    backward_fn = get_backward_compare(self.api_config.api_name)
                    if backward_fn is not None:
                        backward_fn(local_grads, remote_grads, self.api_config, tester=self)
                    elif isinstance(local_grads, (list, tuple)) and isinstance(
                        remote_grads, (list, tuple)
                    ):
                        for i, (local_grad, remote_grad) in enumerate(
                            zip(local_grads, remote_grads)
                        ):
                            if isinstance(local_grad, paddle.Tensor) and isinstance(
                                remote_grad, paddle.Tensor
                            ):
                                dtype_str = str(local_grad.dtype).split(".")[-1]
                                atol, rtol = self._resolve_atol_rtol(dtype_str)
                                local_np = local_grad.numpy()
                                remote_np = remote_grad.numpy()
                                self._print_diff(f"Backward gradient[{i}]", local_np, remote_np)
                                self._assert_close(local_np, remote_np, atol, rtol)
                                print(
                                    f"[compare] Backward gradient[{i}] comparison passed",
                                    flush=True,
                                )
                    elif isinstance(local_grads, paddle.Tensor) and isinstance(
                        remote_grads, paddle.Tensor
                    ):
                        dtype_str = str(local_grads.dtype).split(".")[-1]
                        atol, rtol = self._resolve_atol_rtol(dtype_str)
                        local_np = local_grads.numpy()
                        remote_np = remote_grads.numpy()
                        self._print_diff("Backward", local_np, remote_np)
                        self._assert_close(local_np, remote_np, atol, rtol)

                    print(
                        f"[compare] Backward gradient check passed for {self.api_config.config}",
                        flush=True,
                    )
                except SkipComparison as e:
                    print(
                        f"[compare] Backward skipped for {self.api_config.config}: {e}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[compare] Backward gradient check failed for {self.api_config.config}, error: {e}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return False

            print(
                f"[compare] Accuracy check passed for {self.api_config.config}",
                flush=True,
            )
            write_to_log("pass", self.api_config.config)
            return True

        except Exception as e:
            print(
                f"[compare] Comparison failed for {self.api_config.config}, error: {e}",
                flush=True,
            )
            write_to_log("accuracy_error", self.api_config.config)
            return False

    def test(self):
        """Main test function"""
        if self.operation_mode == "upload":
            self._test_upload_mode()
        elif self.operation_mode == "download":
            self._test_download_mode()
        elif self.operation_mode == "http":
            self._test_http_mode()
        else:
            print(
                "[error] operation_mode 不能为空，请指定 --operation_mode=upload 或 download 或 http",
                flush=True,
            )
            return

    def _test_upload_mode(self):
        """Upload模式：执行测试并上传结果"""
        print(f"[upload] Starting upload mode for {self.api_config.config}", flush=True)

        local_device_type = self._get_local_device_type()
        output, grads = self._run_paddle(local_device_type)

        if output is None:
            print(f"[upload] Execution failed for {self.api_config.config}", flush=True)
            return

        # 保存结果到本地PDTensor
        local_path = self._save_tensor_locally(output, grads)

        # 异步上传到BOS
        self._upload_to_bos(local_path)

        print(f"[upload] Upload mode completed for {self.api_config.config}", flush=True)

    def _test_download_mode(self):
        """Download模式：下载对比数据并验证"""
        print(
            f"[download] Starting download mode for {self.api_config.config}",
            flush=True,
        )

        # 确定要下载的文件名（与 GPU 上传时保持一致）
        target_filename = self._get_filename()

        # 下载文件
        downloaded_file = self._download_from_bos(target_filename)
        if downloaded_file is None:
            print(
                f"[download] Failed to download comparison data for {self.api_config.config}",
                flush=True,
            )
            return

        # 在本地设备上执行测试
        local_device_type = self._get_local_device_type()
        local_output, local_grads = self._run_paddle(local_device_type)

        if local_output is None:
            print(
                f"[download] Local execution failed for {self.api_config.config}",
                flush=True,
            )
            return

        # 与下载的结果进行对比
        success = self._compare_with_downloaded(local_output, local_grads, downloaded_file)

        # 清理下载的文件
        downloaded_file.unlink(missing_ok=True)

        print(
            f"[download] Download mode completed for {self.api_config.config}",
            flush=True,
        )

    def _request_remote_execution(self):
        """发送API配置到远程HTTP服务器执行并返回 (bytes, None) 或 (None, error_info)"""
        import json
        import urllib.error
        import urllib.request

        url = f"http://{self.http_host}:{self.http_port}/run_api_test"
        payload = json.dumps(
            {
                "api_config": self.api_config.config,
                "random_seed": self.random_seed,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                if resp.status == 200:
                    return resp.read(), None
                body = resp.read().decode("utf-8", errors="replace")
                print(f"[http] Remote returned status {resp.status}: {body}", flush=True)
                return None, {"error": "unknown", "detail": body}
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
                error_info = json.loads(body)
            except Exception:
                error_info = {"error": "unknown", "detail": str(e)}
            status = e.code
            error_type = error_info.get("error", "unknown")
            detail = error_info.get("detail", "")
            print(
                f"[http] HTTP error {status} ({error_type}): {detail}",
                flush=True,
            )
            return None, error_info
        except urllib.error.URLError as e:
            print(f"[http] Network error: {e.reason}", flush=True)
            return None, {"error": "network_error", "detail": str(e.reason)}
        except Exception as e:
            print(f"[http] Request failed: {e}", flush=True)
            return None, {"error": "network_error", "detail": str(e)}

    def _save_bytes_to_temp(self, data_bytes):
        """将接收到的字节数据保存到临时文件"""
        temp_dir = tempfile.gettempdir()
        filename = f"http_{self._get_filename()}"
        local_path = Path(temp_dir) / filename
        local_path.write_bytes(data_bytes)
        return local_path

    def _test_http_mode(self):
        """HTTP模式：发送API配置到远程服务器，获取结果并本地对比"""
        # Skip before sending HTTP request (e.g. float8 on XPU)
        if self.need_skip(paddle_only=True):
            write_to_log("skip", self.api_config.config)
            return

        print(f"[http] Starting HTTP mode for {self.api_config.config}", flush=True)

        # 1. 发送到远端执行
        result_bytes, error_info = self._request_remote_execution()

        if result_bytes is None:
            # 根据错误类型写对应日志
            if error_info:
                error_type = error_info.get("error", "unknown")
                if error_type in ("remote_error", "cuda_error", "oom", "crash", "timeout"):
                    write_to_log(error_type, self.api_config.config)
                elif error_type == "skip":
                    write_to_log("skip", self.api_config.config)
                elif error_type == "network_error":
                    write_to_log("network_error", self.api_config.config)
                else:
                    print(
                        f"[http] Unknown remote error for {self.api_config.config}",
                        flush=True,
                    )
            return

        # 2. 保存远端结果到临时文件
        downloaded_tensor_path = self._save_bytes_to_temp(result_bytes)

        # 3. 在本地设备上执行（设置与服务端相同的随机种子，保证输入数据一致）
        np.random.seed(self.random_seed)
        paddle.seed(self.random_seed)
        local_device_type = self._get_local_device_type()
        local_output, local_grads = self._run_paddle(local_device_type)

        if local_output is None:
            print(
                f"[http] Local execution failed for {self.api_config.config}",
                flush=True,
            )
            downloaded_tensor_path.unlink(missing_ok=True)
            return

        # 4. 用现有对比逻辑进行比较
        self._compare_with_downloaded(local_output, local_grads, downloaded_tensor_path)

        # 5. 清理临时文件
        downloaded_tensor_path.unlink(missing_ok=True)

        print(f"[http] HTTP mode completed for {self.api_config.config}", flush=True)
