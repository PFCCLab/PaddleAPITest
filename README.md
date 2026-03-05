# PaddleAPITest

## 1. 项目背景

正确性是 PaddlePaddle 框架质量的基石，影响业务训练、推理的效果，影响用户对 Paddle 的信赖。如何主动发现 Paddle 存在的质量问题并及时修复，而不是被动等用户反馈后再修复，是一个亟待解决的质量难题。

API 是 Paddle 的 “窗口”，PaddleAPITest 通过整理超过 300 万条 API 级别的测试用例（本项目称为配置 / api config），利用这些用例驱动 Paddle 的不同内核机制执行 API，从而成为 Paddle 内核机制和算子正确性的 “扫描仪”。

PaddleAPITest 主要工作思路如下：

1. 在 Paddle 中开发 Trace API 机制（具体实现见 [PR#70752](https://github.com/PaddlePaddle/Paddle/pull/70752)），用于抓取 API 调用配置。以下是一个配置示例：
```
paddle.concat(tuple(Tensor([31376, 768],"float32"),Tensor([1, 768],"float32"),), axis=0, )
```

2. 在所有 Paddle 单元测试（CI）和集成测试（CE）流水线中，抓取所有 Paddle API 的调用配置，形成了 `PaddleAPITest/tester/api_config` 下的 `CI_CE_config` 的配置集。对配置集进行去重、排序，并对测试结果进行梳理，得到了 `api_config` 下各个目录的配置集。

3. 在 PaddleAPITest 中开发一套 **引擎**，用于加载配置集，初始化相应 Tensor，并调用相应的 API 执行前/反向测试。

4. 对采集到的配置集进行 shape 篡改，生成了以 `"big_tensor"` 和 `"0_size"` 开头的配置集。

5. 对于精度正确性，在 PaddleAPITest 中开发一套 **转换工具**，在调用 Paddle API 测试的同时，等效地调用 Torch API，进行精度对比测试。

6. 对于内核测试，可通过继承 `APITestBase`，开发针对特定内核的测试引擎。测试对象包括：Kernel 精度、Kernel 性能、Kernel 显存、动态图、静态图、动转静、组合算子、CINN、Paddle2ONNX、GPU、CPU、XPU、NPU、OneDNN、TRT 等等。

## 2. 项目结构

```bash
├── report/
├── test_pipline/
├── tester/
│   ├── api_config/
│   ├── paddle_to_torch/
│   ├── accuracy.py
│   ├── base.py
│   ├── paddle_cinn_vs_dygraph.py
│   └── paddle_only.py
├── tools/
├── engine.py
├── engineV2.py
├── engineV3.py
└── run-example.sh
```

项目结构主要分为 `report` 和 `tester` 文件夹，`report` 用于存储内核报错的 api 信息，`tester` 用于测试配置的正确性和存放配置测试结果。

**`engineV2.py`** 及配套的 `run-example.sh` 是目前运行本项目的主要工具；**`engineV3.py`** 目前由百度内部开发测试使用；**`engine.py`** 是最早的引擎，相较 `engineV2.py` 吞吐量低，在少量配置时可使用。

1. report 介绍
   - 0size_tensor_cpu 存放进行在 cpu 上进行精度测试/引擎解析能力测试（accuracy / paddle_only）结果。
   - 0size_tensor_gpu 存放进行在 gpu 上进行 accuracy / paddle_only 测试结果。
   - big_tensor_gpu 存放大形状张量（big tensor）在 gpu/cpu 上进行 accuracy / paddle_only 测试结果。
   - ci_ce_cpu 存放 CI/CE 流水线抓取的配置在 cpu 上进行 accuracy / paddle_only 测试结果。
   - ci_ce_gpu 存放 CI/CE 流水线抓取的配置在 gpu 上进行 accuracy / paddle_only 测试结果。
   - cinn 存放 paddle 静态编译器与动态图方式进行精度对比测试结果。
   - fresh_report 存放引擎补齐（paddle_only）和精度转换（accuracy）两个任务中，出现的内核报错或者精度报错。

2. tester 介绍

   * api_config 目录存放配置目录的管理情况和相关脚本工具，各配置目录下的文本命名语义一致，参考 5_accuracy 中的 txt 命名含义：
     * 1_not_support 目录下的配置为 paddle 不支持的配置，例如数据类型对应的算子未实现。
     * 2_paddle_only_random 为 PaddleAPITest 引擎支持对配置的解析（--paddle_only=True），但是配置输出随机，也就无法继续进行精度测试的配置。random_creation.txt 为创建随机数的配置，random_calculation.txt 为具有随机行为的函数。
     * 3_paddle_only 支持引擎解析（本项目称为 merged），但是无法进行精度测试（也称为 merged_not_support）的配置。
     * 4_paddle_only_amp 在混合精度模式下支持引擎解析的配置（--paddle_only=True --test_amp=True）。
     * 5_accuracy 支持精度测试的配置，不同的命名对应不同的测试结果：
       * accuracy_*.txt 为测试通过。
       * accuracy_cpu_error.txt 为 **在 cpu 上运行精度测试不通过（--test_cpu=True）** 的配置集。
       * accuracy_cpu_kernel.txt 为 **paddle 内核抛出错误** 的配置集。
       * accuracy_gpu_error.txt 为 **在 gpu 上运行精度测试不通过** 的配置集。
       * accuracy_gpu_error_dtype_diff.txt 为回归测试中发现精度错误，但是检查精度时 **不强制对齐 dtype 后能够通过** 的配置（base.py 中的 not_check_dtype 列表中的 api）。
       * accuracy_gpu_error_grads_diff.txt 为 **paddle 和 torch api 反向梯度结果不同，无法进行比较，输出 [not compare]** 的配置集。
       * accuracy_gpu_error_uncertain.txt 为 **不确定是因为 PaddleAPITest 引擎转化能力 还是 Paddle 内核实现** 导致的 accuracy error 的配置集。
     * 6_accuracy_amp 类似 5_accuracy，只是需要混合精度运行测试（--test_amp=True）的配置集。
     * 7_0_size 张量形状含有 0（0-size）的配置集。
     * 8_big_tensor 对配置的张量形状基于 to_big_size_config.py 进行篡改的配置集。
     * 9_getset_item 为测试 paddle.Tensor.\_\_getitem__ 和 paddle.Tensor.\_\_getitem__ 使用的配置集。
     * big_and_0size 为含有大形状张量和 0-size 张量使用的配置集。
     * CI_CE_config 为 CI，CE 抓取的配置集。
     * 脚本工具
       * config_analyzer.py 是引擎对配置解析并针对 api 初始化合适张量的代码（引擎补齐任务产物）
       * log_writer.py 是 engine 写入日志的工具
       * to_0_size*.py 是篡改为 0-size 配置的工具
       * to_big_size\*.py 是篡改为大形状张量的配置的工具。 

   * tester/paddle2torch/ 是转换能力的核心代码。介绍详见 [4. paddle2torch转换](#4-paddle2torch转换)

3. tools 文件夹中存放了一些实用的工具，例如 move_config.py 可以用来批量的移动配置，error_stat.py 可以一键解析错误日志等。

## 3. 使用介绍

### 3.1 环境配置

1. 建议在虚拟环境或 docker 中进行开发，并正确安装 python 与 nvidia 驱动。

2. PaddlePaddle 框架运行环境分为 **CPU** 环境与 **GPU** 环境，CPU 和 GPU 上运行的结果 **可能存在差异**，即存在 GPU 上能够正确运行，但 CPU 上报错的情况。请正确安装 *paddlepaddle-gpu* 环境，选择 develop 版本：
   - [使用 pip 快速安装 paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
   ```
   - 若需要本地编译 Paddle，可参考链接：[Linux 下使用 ninja 从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile-by-ninja.html)

3. 安装 PaddleAPITest 项目其他依赖项：
   - [使用 pip 快速安装 torch](https://pytorch.org/get-started/locally/)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   - 安装第三方库：
   ```bash
   pip install func_timeout pandas pebble pynvml pyyaml typer
   ```

4. PaddlePaddle 与 PyTorch 的部分依赖项可能发生冲突，请先安装 *paddlepaddle-gpu* 再安装 *torch*，重新安装请在 pip 后添加 `--force-reinstall` 参数，仅更新 paddle 请添加 `--no-deps` 参数；engineV2 建议使用 python>=3.10

### 3.2 使用说明

#### A. engineV1

测试时，`tester/api_config/test_log` 文件夹用于存放测试所产生的测试结果和 checkpoint。

PaddleAPITest 目前支持 `paddle_only`、`accuracy`、`paddle_cinn` 三种测试：

- **paddle_only**，用于单独将 Paddle 动态图跑一遍，验证 paddle 框架以及 PaddleAPITest 引擎**是否支持**该配置。
- **accuracy**，用于将 Paddle API 的前反向与 **Torch** 的前反向做精度对比测试。
- **paddle_cinn**，用于将 Paddle 动态图与 Paddle 静态图编译器做精度对比测试。
- **paddle_custom_device**, 用于将Custom Device或者XPU的API与CPU做精度对比。
当测试**单个配置**时，可使用下面的代码，`--api_config` 中输入待测试的配置内容：

- 仅测试 paddle **是否支持**：
```bash
python engine.py --paddle_only=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```

- 测试输出**是否准确**：
```bash
python engine.py --accuracy=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```
- 动态图和静态图测试：

```bash
python engine.py --paddle_cinn=True --api_config='paddle.abs(Tensor([1, 100],"float64"), )'
```

> [!NOTE]
> **注意**: 配置 txt 中统一使用双引号 `"`，因此建议 `--api_config=''` 使用单引号，或在配置中手动添加转义斜杠 `\`

当需要测试的配置数目较多时，手动单次输入将**非常低效**，这种情况下可以使用如下所示的**批量测试**指令，将配置保存在一个 txt 中，并将指令中的路径设置为 txt 的路径即可：
```bash
python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --accuracy=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_only=True > tester/api_config/test_log/log.log 2>&1

python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config.txt --paddle_cinn=True > tester/api_config/test_log/log.log 2>&1
```

当测试配置中有**精度不统一**的情况，需要精度转换时，直接运行测试可能会报错，可加入`--test_amp=True`。  

如果希望在引发`paddle_error`或者`accuracy_error` 错误时退出测试，可加入`--exit_on_error=True`,此时测试进程`exit_code`为1。

#### B. engineV2

`engineV2.py` 是为 PaddleAPITest 项目设计的高性能测试框架，支持多 GPU 并行执行，具备负载均衡、超时处理和崩溃恢复能力。相比原始的 `engine.py` 实现，它能显著提升 Paddle API 配置测试效率，加速比约为 5-10 倍。

功能特性:
- **多 GPU 并行**：拥有灵活的 *gpus 相关参数*，可配置多 GPU 并行测试，支持任务跨 GPU 动态分发
- **进程级并行**：基于 Pebble 库的 ProcessPool 进程池实现，支持进程的高效并行，每张 GPU 可拥有多个 worker
- **动态负载均衡**：新的子进程自动分配至负载最轻 GPU，计算资源最优利用
- **超时/崩溃恢复**：由张量大小推断执行时限（梯度阈值），主动杀死 coredump 进程，自动检测并重启死亡进程

以精度测试为例，配置文件路径为 `tester/api_config/api_config_tmp.txt`，输出日志路径为 `tester/api_config/test_log`：

**多 GPU 多进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" >> "tester/api_config/test_log/log.log" 2>&1
```

**多 GPU 单进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --gpu_ids="0,1" >> "tester/api_config/test_log/log.log" 2>&1
```

**单 GPU 单进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --num_gpus=1 >> "tester/api_config/test_log/log.log" 2>&1
```

**使用 run.sh 脚本**：
```bash
# chmod +x run.sh
./run.sh
```
该脚本使用参数：`NUM_GPUS=-1, NUM_WORKERS_PER_GPU=-1, GPU_IDS="4,5,6,7"`，在后台运行程序，可在修改 `run.sh` 参数后使用。说明文档详见：[engineV2-README.md](engineV2-README.md)

#### C. engineV3

暂无介绍

## 4. paddle2torch转换

Paddle2Torch 是一个专注于将 PaddlePaddle API 转换为 PyTorch 对应实现的知识工具库，属于 [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest) 项目的核心组成模块。本模块通过解析 PaddlePaddle API 调用，使用预定义的转换规则与动态代码生成，实现从 PaddlePaddle 到 PyTorch 的自动转换。转换过程将确保代码的语义一致性。

本模块具有精简强悍的架构，仅由三个组件构成：
- *转换引擎 converter.py*
- *转换配置 mapping.json*
- *转换规则 rules.py*

代码已完全进行解耦，可以非常容易地迁移至其他代码中。本模块通过 **转换配置** 与 **转换规则** 管理 API 映射关系，因此支持开发者灵活扩展新的 API 转换能力。

本模块的典型应用场景包括：模型迁移、跨框架验证、混合编程等，可为深度学习开发者提供跨框架的互操作性解决方案。现在转换工具已基本完成对PaddleAPI的转换。说明文档详见：[paddle_to_torch/README.md](tester/paddle_to_torch/README.md)

> [!TIP]
> 本 README 已经过 ***文心一言 4.5 Turbo*** 润色
