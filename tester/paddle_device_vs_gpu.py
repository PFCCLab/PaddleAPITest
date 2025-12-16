from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import paddle

from .api_config.log_writer import write_to_log
from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU


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

    def _run_paddle(self, device_type: str):
        """在指定设备上运行 Paddle（统一 GPU / XPU / 自定义设备逻辑）。"""
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

            paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)

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

            return paddle_output, paddle_grads

        except Exception as e:
            print(
                f"[paddle {paddle_device_type} error] {self.api_config.config}: {e}",
                flush=True,
            )
            write_to_log("paddle_error", self.api_config.config)
            return None, None

    def _compare_with_downloaded(self, local_output, local_grads, downloaded_tensor):
        """与下载的结果进行对比"""
        try:
            print(f"[compare] Comparing results for {self.api_config.config}", flush=True)

            # 加载下载的数据
            remote_data = paddle.load(str(downloaded_tensor))
            remote_output = remote_data["output"]

            # 对比Forward输出（直接使用Paddle对比）
            try:
                if isinstance(local_output, paddle.Tensor) and isinstance(
                    remote_output, paddle.Tensor
                ):
                    # 使用Paddle的对比方法
                    np.testing.assert_allclose(
                        local_output.numpy(),
                        remote_output.numpy(),
                        atol=self.atol,
                        rtol=self.rtol,
                        equal_nan=True,
                    )
                elif isinstance(local_output, (list, tuple)) and isinstance(
                    remote_output, (list, tuple)
                ):
                    # 列表或元组对比
                    for i, (local_item, remote_item) in enumerate(zip(local_output, remote_output)):
                        if isinstance(local_item, paddle.Tensor) and isinstance(
                            remote_item, paddle.Tensor
                        ):
                            np.testing.assert_allclose(
                                local_item.numpy(),
                                remote_item.numpy(),
                                atol=self.atol,
                                rtol=self.rtol,
                                equal_nan=True,
                            )
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
                    if isinstance(local_grads, (list, tuple)) and isinstance(
                        remote_grads, (list, tuple)
                    ):
                        for i, (local_grad, remote_grad) in enumerate(
                            zip(local_grads, remote_grads)
                        ):
                            if isinstance(local_grad, paddle.Tensor) and isinstance(
                                remote_grad, paddle.Tensor
                            ):
                                np.testing.assert_allclose(
                                    local_grad.numpy(),
                                    remote_grad.numpy(),
                                    atol=self.atol,
                                    rtol=self.rtol,
                                    equal_nan=True,
                                )
                                print(
                                    f"[compare] Backward gradient[{i}] comparison passed",
                                    flush=True,
                                )
                    elif isinstance(local_grads, paddle.Tensor) and isinstance(
                        remote_grads, paddle.Tensor
                    ):
                        np.testing.assert_allclose(
                            local_grads.numpy(),
                            remote_grads.numpy(),
                            atol=self.atol,
                            rtol=self.rtol,
                            equal_nan=True,
                        )

                    print(
                        f"[compare] Backward gradient check passed for {self.api_config.config}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[compare] Backward gradient check failed for {self.api_config.config}, error: {e}",
                        flush=True,
                    )
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
        else:
            print(
                "[error] operation_mode 不能为空，请指定 --operation_mode=upload 或 download",
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
