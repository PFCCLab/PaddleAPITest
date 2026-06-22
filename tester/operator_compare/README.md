# Operator Compare

`tester/operator_compare` 复用 PaddleAPITest config、`TensorConfig` 和 Paddle→Torch converter，对指定算子 config 进行 Paddle / Torch / custom 多实现数值对比，并生成结构化结果与报告。

## 1. 使用场景

用于在开发、调试或扩展算子时，对单个或少量 API config 做多实现数值对比、误差分析和报告归档。

## 2. 功能简介

### 2.1 执行流程

```text
PaddleAPITest config txt
        │
        ▼
APIConfig / TensorConfig 解析输入
        │
        ▼
CompareSuite 展开 implementation × dtype × precision
        │
        ▼
runner 执行 paddle / torch / custom 实现
        │
        ▼
metrics 计算 standard 与 pairwise 误差
        │
        ▼
artifacts / report 写出 JSON、CSV、Markdown 和 figures
```

### 2.2 核心能力

- **复用 config**：直接读取 PaddleAPITest config 行或 config txt 文件。
- **复用输入生成**：通过 `APIConfig` / `TensorConfig` 构造 Paddle 和 Torch 共用输入。
- **复用 Torch reference**：通过 `Paddle2TorchConverter` 将 Paddle API 映射到 Torch 实现。
- **多实现对比**：支持 `paddle`、`torch` 和注册的 custom implementation。
- **矩阵展开**：支持按 implementation、dtype、precision 展开比较。
- **统一输出**：生成 `results.json`、`summary.csv`、pairwise CSV、`report.md` 和可选图示。

### 2.3 目录结构

```text
tester/operator_compare/
├── spec.py                       # CompareCase / ImplementationSpec / CompareSuite
├── config_loader.py              # config txt -> CompareCase
├── implementations.py            # paddle/torch/custom 实现声明、参数绑定、dtype matrix 展开
├── runner.py                     # 多实现执行、standard 和 pairwise metrics 调度
├── metrics.py                    # 误差指标和 tensor fingerprint
├── artifacts.py                  # env/results/summary/pairwise 写出
├── report.py                     # Markdown 报告和图示
└── profile.py                    # profile 状态和 sqlite kernel summary 解析
```

工具入口：

```text
tools/operator_compare.py
tools/operator_compare_profile_worker.py
```

## 3. 开发指南

### 3.1 优先方式：只增加 config

适用于：

- Paddle API 能直接执行。
- Torch reference 已在 `tester/paddle_to_torch/mapping.json` / `rules.py` 中支持。
- 输出可以由通用输出处理转换为 `torch.Tensor`。

示例：

```bash
python tools/operator_compare.py \
  --case 'paddle.add(Tensor([2], "float32"), Tensor([2], "float32"), )' \
  --implementations paddle,torch \
  --standard 'torch|config|default' \
  --output-dir test_log_operator_compare/add_smoke
```

批量覆盖 shape / dtype 时，把多条 config 写入 txt，再通过 `--config-file` 运行。

### 3.2 Torch reference 不存在：补充 converter rule

如果 `Paddle2TorchConverter.convert(api_name)` 不支持目标 API，需要补充 Paddle→Torch 映射。

简单映射优先在 `tester/paddle_to_torch/mapping.json` 中使用 `GenericRule` 字段：

```json
"paddle.add": {
  "torch_api": "torch.add",
  "paddle_torch_args_map": {
    "x": "input",
    "y": "other"
  },
  "torch_kwargs": {
    "alpha": 1
  }
}
```

如果需要特殊计算逻辑，在 `tester/paddle_to_torch/rules.py` 增加专用 rule，并在 `mapping.json` 中配置：

```json
"paddle.xxx": {
  "Rule": "YourRule"
}
```

### 3.3 `_C_ops` 无签名算子：补充参数名绑定

部分 `_C_ops` 没有 Python signature，而 converter rule 可能通过 `locals().get("x")` 等命名变量取参数。此时在 `tester/operator_compare/implementations.py` 的 `MANUAL_ARGUMENT_NAMES` 中补充位置参数名：

```python
MANUAL_ARGUMENT_NAMES: dict[str, tuple[str, ...]] = {
    "paddle._C_ops.fused_linear_param_grad_add": (
        "x",
        "dout",
        "dweight",
        "dbias",
        "multi_precision",
        "has_bias",
    ),
}
```

要求：

1. key 使用完整 API name。
2. tuple 顺序与 config 中位置参数顺序一致。
3. 名称与 converter rule 中读取的 locals 名称一致。
4. 仅在 `inspect.signature()` 无法绑定时增加手工映射。

### 3.4 通用实现无法表达：注册 custom implementation

当 `paddle` / `torch` 通用路径无法表达某个实验实现时，在 `tester/operator_compare/implementations.py` 中注册 custom runner：

```python
from tester.operator_compare.implementations import register_custom_implementation


def my_runner(case, spec):
    api_config = case.tensors["api_config"]
    ...
    return torch_tensor


register_custom_implementation("my_impl", my_runner)
```

runner 接口：

```python
def runner(case: CompareCase, spec: ImplementationSpec) -> torch.Tensor:
    ...
```

要求：

- 返回值必须是 `torch.Tensor`。
- 尽量复用 `case.tensors["api_config"]` 中的 `APIConfig` / `TensorConfig`。
- custom 名称通过 `--implementations paddle,torch,my_impl` 使用。

### 3.5 特殊输出处理

通用输出处理在 `to_torch_tensor()` 中完成：

- `torch.Tensor`：直接返回。
- `paddle.Tensor`：通过 DLPack 转成 Torch tensor。
- `list` / `tuple`：取第一个 tensor 输出。

如果新算子的有效输出不是第一个 tensor，优先在 `implementations.py` 中集中补充输出选择逻辑，并增加对应测试。

## 4. 测试指南

新增算子时建议覆盖：

1. config 能被 `APIConfig` 正确解析。
2. `build_compare_suite()` 能生成预期 implementation id。
3. Paddle / Torch 两个实现都能运行成功。
4. target 相对 standard 的误差符合预期。
5. 如果新增 `_C_ops` 参数绑定，使用对应 `_C_ops` config 做 CLI smoke。
6. 如果新增 custom implementation，使用包含 custom implementation id 的 CLI smoke 验证。

语法检查：

```bash
python -m py_compile \
  tools/operator_compare.py \
  tools/operator_compare_profile_worker.py \
  tester/operator_compare/__init__.py \
  tester/operator_compare/artifacts.py \
  tester/operator_compare/config_loader.py \
  tester/operator_compare/implementations.py \
  tester/operator_compare/metrics.py \
  tester/operator_compare/profile.py \
  tester/operator_compare/report.py \
  tester/operator_compare/runner.py \
  tester/operator_compare/spec.py
```

真实执行 smoke 示例见下文 `add smoke` 和 `fused_linear_param_grad_add smoke`。

## 5. 使用示例

### 5.1 单条 config

```bash
python tools/operator_compare.py \
  --case 'paddle.Tensor.__abs__(Tensor([], "float32"), )' \
  --implementations paddle,torch \
  --standard 'torch|config|default' \
  --output-dir test_log_operator_compare/abs_smoke
```

### 5.2 多条 config 文件

```bash
python tools/operator_compare.py \
  --config-file tester/api_config/5_accuracy/example.txt \
  --op paddle.Tensor.__abs__ \
  --implementations paddle,torch \
  --output-dir test_log_operator_compare/abs_file
```

### 5.3 指定 dtype matrix

```bash
python tools/operator_compare.py \
  --case 'paddle.Tensor.__abs__(Tensor([2], "float32"), )' \
  --implementations paddle,torch \
  --dtypes float32,float64 \
  --standard 'torch|float32|default'
```

### 5.4 `add` smoke

```bash
python tools/operator_compare.py \
  --case 'paddle.add(Tensor([2], "float32"), Tensor([2], "float32"), )' \
  --implementations paddle,torch \
  --standard 'torch|config|default' \
  --output-dir test_log_operator_compare/add_smoke_flat
```

### 5.5 `fused_linear_param_grad_add` smoke

```bash
python tools/operator_compare.py \
  --case 'paddle._C_ops.fused_linear_param_grad_add(Tensor([8, 4], "float32"), Tensor([8, 4], "float32"), Tensor([4, 4], "float32"), None, False, False, )' \
  --implementations paddle,torch \
  --standard 'torch|config|default' \
  --output-dir test_log_operator_compare/fused_linear_param_grad_add_smoke_flat
```

`--standard` 中包含 `|` 时需要加引号。

## 6. 参数与输出

### 6.1 常用参数

| 参数 | 说明 |
| --- | --- |
| `--case` | 单条 PaddleAPITest config 字符串。 |
| `--config-file` | PaddleAPITest config txt 文件。 |
| `--op` | 可选 API name 过滤，用于 config 文件。 |
| `--implementations` | 逗号分隔实现，默认 `paddle,torch`。 |
| `--dtypes` | 逗号分隔 dtype 覆盖矩阵；不指定时使用 config 原 dtype。 |
| `--precisions` | 逗号分隔精度策略名，默认 `default`。 |
| `--standard` | 标准实现 id，例如 `torch|config|default`。 |
| `--metrics-dtype` | 指标计算 dtype，支持 `fp32` / `fp64`。 |
| `--no-fingerprint` | 关闭输出 tensor SHA256 fingerprint。 |
| `--output-dir` | 输出目录。 |

### 6.2 输出文件

```text
env.json                         # Python/Paddle/Torch/CUDA/env/config 信息
results.json                     # 完整结构化结果
summary.csv                      # 每个实现相对 standard 的误差摘要
pairwise_summary.csv             # target vs reference pairwise 误差
reference_pairwise_summary.csv   # reference 内部 pairwise 误差
report.md                        # Markdown 报告
figures/                         # 图示，依赖 matplotlib/numpy
```
