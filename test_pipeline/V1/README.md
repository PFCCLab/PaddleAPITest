# 测试流水线工具集

此目录包含用于支持 PaddleAPITest 项目中不同测试流水线的各种脚本和工具。这些工具通常自动化诸如测试分割、日志分析、结果聚合和清理等任务，从而为更复杂或大规模的测试场景提供便利。

**目录**

*   [测试准备](#测试准备)
*   [目录结构](#目录结构)
*   [工具详解](#工具详解)
*   [使用方法](#使用方法)
*   [总结](#总结)

---
## 测试准备

为确保测试结果的准确性与时效性，请务必在测试开始前完成 PaddlePaddle 最新开发环境的配置。

*   **GPU 版本测试**：
    请访问 [PaddlePaddle GPU每日构建版本链接](https://www.paddlepaddle.org.cn/packages/nightly/cu118/paddlepaddle-gpu/) 下载并安装最新的 `wheel` 包。

*   **CPU 版本测试**：
    请执行以下命令安装 PaddlePaddle：
    ```bash
    python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
    ```
    **注意**：目前 CPU 版本的 PaddlePaddle 并非每日更新。安装完成后，建议通过执行 `paddle.__git_commit__` 命令核实当前所安装版本的具体 commit id。


## 目录结构

`test_pipeline` 目录旨在组织针对不同测试目标、环境或类型的辅助脚本和配置文件。每个子目录代表一个特定的测试场景。这种结构有助于保持相关工具的集中和清晰。

以下是当前的子目录及其用途说明：
*   **`cpu_0size/`**: <a name="cpu_0size"></a>
    *   **用途**: 存放 CPU 环境下处理 PaddlePaddle API 0-size tensor 报错信息的python脚本。

*   **`gpu_0size/`**: <a name="gpu_0size"></a>
    *   **用途**: 存放与在 GPU 环境下测试 PaddlePaddle API 处理 0-size tensor 相关的工具和脚本。
    *   **子目录**:
        *   `gpu_0size_accuracy/`: 包含专门用于 **管理和分析 GPU 上 0-size tensor accuracy 测试** 结果的脚本。
        *   `gpu_0size_paddleonly/`: 包含专门用于 **管理和分析 GPU 上 0-size tensor paddle_only 测试** 结果的脚本。
*   **`gpu_bigtensor/`**: <a name="gpu_bigtensor"></a>
    *   **用途**: 存放与在 GPU 环境下测试 PaddlePaddle API 处理 大 tensor 相关的工具和脚本。
    *   **子目录**:
        *   `gpu_bigtensor_accuracy/`: 包含专门用于 **管理和分析 GPU 上 big tensor accuracy 测试** 结果的脚本。
        *   `gpu_bigtensor_paddleonly/`: 包含专门用于 **管理和分析 GPU 上 big tensor paddle_only 测试** 结果的脚本。
*   **`gpu_paddleonly/`**: <a name="gpu_paddleonly"></a>
    *   **用途**: 存放与在 GPU 环境下对常规 API 进行 `paddle_only=True` 测试相关的工具和脚本。这种测试模式用于验证 API 在仅使用 PaddlePaddle 内部实现（不涉及与其他框架的对比）时的行为和稳定性。
    *   **结构**: 直接包含执行该测试场景所需的各类工具脚本（如 `split.sh`, `run*.sh`, `error_stat.py`, `rm_*.sh` 等）。
*   **`cpu_accuracy/`**: <a name="cpu_accuracy"></a>
    *   **用途**: 存放与在 CPU 环境下对常规 API 进行 `accuracy=True` 测试相关的工具和脚本。
    *   **结构**: 直接包含执行该测试场景所需的各类工具脚本（如 `split.sh`, `run*.sh`, `error_stat.py`, `rm_*.sh` 等）。

## 工具详解

`test_pipeline/gpu_0size/gpu_0size_accuracy/` 目录下的脚本是此类工具的代表例子。它们用于管理和分析在 GPU 上测试 0-size tensor 精度的结果，下面以该目录为示例，对文件内脚本进行讲解。

#### Shell 脚本 (`.sh`)

这些脚本通常处理文件操作、测试执行编排和清理工作。

*   `split.sh`: 读取主测试列表文件 (`0sizetensor_accuracy.txt`)，并将其分割成指定数量（在此示例中为 24 个）的较小文件 (`0size_tensor_accuracy_*.txt`)。
*   `gpu_0size_accuracy_run[1-24].sh`: 执行测试的单元脚本，每个脚本负责运行一个测试子集。
*   `start_all.sh`: 用于**批量启动**所有 `gpu_0size_accuracy_run*.sh` 脚本。
*   `stop_all.sh`: 用于**批量停止**所有由 `start_all.sh` 启动的测试进程。
*   `rm_log.sh`: 清理运行脚本生成的日志文件 (`0size_tensor_accuracy_*.log`)。
*   `rm_testlog.sh`: 清理通用的测试日志目录 (`tester/api_config/test_log/*`)。
*   `rm_checkpoint.sh`: 删除检查点文件 (`tester/api_config/test_log/checkpoint.txt`)。
*   `rm_all.sh`: 全面的清理脚本，用于删除 `gpu_0size_accuracy` 的所有相关日志。

#### Python 脚本 (`.py`)

这些脚本通常用于更复杂的数据处理、分析和基于测试输出的报告生成。

*   `stat.py`: 读取测试列表和日志文件，计算并打印每个测试分片的统计信息。
*   `error_stat.py`: 解析日志文件以识别和汇总错误，生成错误报告文件。

## 使用方法

以下是使用特定测试场景（以 `gpu_0size/gpu_0size_accuracy` 为例，其他目录如 `gpu_paddleonly` 使用方法类似）下工具的一般流程：

1.  **准备测试文件**:
    *   **如果只有一个大的测试列表文件** (例如 `0sizetensor_accuracy.txt`)：
        *   运行 `split.sh` 脚本将其分割成多个小文件: `bash split.sh`
        *   这会生成类似 `<前缀>_1.txt`, `<前缀>_2.txt`, ... 的文件。
    *   **如果已经有多个测试文件**：可以编写统计分割脚本，或直接进行测试。

2.  **编写运行脚本 (`run*.sh`)**:
    *   确保每个测试文件 (例如 `<前缀>_1.txt`) 都有一个对应的运行脚本 (例如 `<场景>_run1.sh`)。
    *   打开每个 `run*.sh` 文件，检查或设置以下内容：
        *   **指定 GPU**: 使用 `export CUDA_VISIBLE_DEVICES=N` 来指定该脚本应在哪块 GPU 卡上运行 (N 是 GPU 的索引，如 0, 1, 2...)。可以根据需要将不同的脚本分配到不同的 GPU 卡上。
        *   **测试命令**: 确保包含正确的测试执行命令。**注意：该命令应在项目根目录 (`PaddleAPITest/`) 下执行。** 命令通常包含一个循环，并调用 `engine.py`。

3.  **执行测试**:
    *   **方式一**:
        *   进入包含 `start_all.sh` 的目录 (例如 `test_pipeline/gpu_0size/gpu_0size_accuracy/`)。
        *   运行 `bash start_all.sh`。这通常会在后台并行启动所有 `run*.sh` 脚本。
    *   **方式二**:
        *   为每个 `run*.sh` 脚本打开一个单独的终端。
        *   在每个终端中，进入项目根目录 (`PaddleAPITest/`)。
        *   运行对应的脚本。

4.  **停止测试**:
    *   如果使用了 `start_all.sh` 启动，并且提供了 `stop_all.sh`，可以进入相应目录运行 `bash stop_all.sh` 来尝试停止所有相关进程。
    *   如果是手动启动的，需要手动 `kill` 对应的进程。

5.  **分析结果**:
    *   等待所有 `run*.sh` 脚本执行完成（日志文件不再增长）。
    *   运行 `python stat.py` 查看每个测试分片的执行统计信息（总数、执行数、通过数等）。
    *   运行 `python error_stat.py` 来提取和汇总错误信息。**这一步非常重要**，它会生成：
        *   `error_log.log`: 包含详细错误信息的日志。
        *   `error_config.txt`: 出错的测试配置列表。
        *   `error_api.txt`: 出错的 API 列表。
        这些文件是后续分析和修复问题的关键依据。

6.  **清理**:
    *   根据需要运行相应的 `rm_*.sh` 脚本来删除生成的日志文件或其他临时文件。例如，运行 `bash rm_log.sh` 删除运行日志，运行 `bash rm_all.sh` 进行更全面的清理。

---

## 总结

`test_pipeline` 目录提供了一套用于管理和执行特定 PaddlePaddle API 测试场景的标准化工具和流程。通过使用这些脚本，开发者可以方便地：

1.  **分割大型测试集**：使用 `split.sh` 将测试任务分解，便于并行处理。
2.  **配置并执行测试**：利用 `run*.sh` 脚本为每个测试子集指定运行环境（如 GPU 卡）并执行测试引擎。
3.  **批量管理测试**：通过 `start_all.sh` 和 `stop_all.sh`（如果提供）简化大量并行测试的启动和停止。
4.  **分析测试结果**：使用 `stat.py` 获取整体执行概况，并利用 `error_stat.py` 精确提取错误信息，为调试和问题定位提供关键输入。
5.  **保持环境整洁**：通过各种 `rm_*.sh` 脚本清理测试过程中产生的日志和临时文件。

遵循“使用方法”部分描述的流程，可以有效地利用这些工具来完成复杂的测试任务，并为后续的错误分析和修复工作打下良好基础。当为新的测试场景添加工具时，建议参考现有目录的结构和脚本功能进行组织和编写。
