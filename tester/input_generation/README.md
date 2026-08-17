# 输入生成运行时

这个目录负责一件事：**根据 API 配置，为每个 Tensor 生成一次可复现的输入值，并在需要时把它物化成 Paddle 或 PyTorch Tensor。**

输入生成只处理输入，不负责执行 API、比较结果或写测试报告。API 专属的输入约束集中在
`generation_rules.py`，所以遇到一个 API 的特殊输入行为时，应先从这个文件开始查找。

## 一次生成做什么

```text
APIConfig
  -> binding.py         找到参数名对应的 TensorConfig 和稳定路径
  -> generation_rules.py 根据 API 语义选择生成规则
  -> values.py          暂存 path、spec 和生成值
  -> backend.py         用 NumPy/Torch/Paddle 原语生成数组或 Tensor
  -> tensor_config.py   保存配置和已物化的 Tensor 缓存
  -> materialization.py 估算物化过程的显存和临时空间
```

规则执行成功后，生成值才会一次性挂回 `APIConfig`。规则中途失败时，不应留下半成品输入。

## 文件职责

### `binding.py`：把输入参数绑定到 Tensor

- 消费 `tester/parameter_binding.py` 的调用绑定结果，识别嵌套容器中的 Tensor。
- 为每个 Tensor 创建稳定的 `InputTensorPath`，例如 `args[0]`、`kwargs.x`、`args[1][2]`。
- 检查同一个 `TensorConfig` 是否被多个路径重复引用。
- 不生成随机值，也不创建 Paddle/PyTorch Tensor。

### `parameter_binding.py`：Paddle 调用参数绑定

- 统一解析 API、手工 C-op 契约、默认值、variadic shape 和 Tensor receiver。
- 执行转换与输入生成共享同一套参数绑定结果，避免各自解释签名。

### `values.py`：输入值对象和路径读写

这里是输入值相关对象的唯一归属：

- `InputTensorPath`：Tensor 在一次 API 调用中的位置。
- `InputTensorSpec`：从 `TensorConfig` 提取的只读规格，包含 shape、dtype、place 和布局信息。
- `InputValue`：规则生成的逻辑值、来源 backend 以及对应路径。
- 路径读取、值挂载、值查询和清理函数。

`InputValue` 是规则提交后的逻辑真源；框架 Tensor 在 materialization 阶段创建。

### `tensor_config.py`：可变配置和 Tensor 缓存

`TensorConfig` 保存 shape、dtype、place、stride 等配置，以及物化后的框架 Tensor 缓存。
它还负责复制、元素数量和单个配置的字节数计算。

它不负责 API 特殊规则，也不负责整棵参数树的物化预估。

### `materialization.py`：物化计划和资源统计

这里处理与“怎样占用资源”有关的逻辑：

- 遍历参数树中的唯一 `TensorConfig`；
- 估算生成值、目标 Tensor 和临时 Tensor 的字节数；
- 创建 `MaterializationPlan`；
- 汇总整棵配置树的元素数量和字节数。

这些函数只根据配置和运行策略计算，不应为了预估显存而提前分配真实 Tensor。

### `backend.py`：三个独立的值生成实现

`InputBackend` 是最小能力 `Protocol`。当前有三个互相独立的实现：

- `NumPyInputBackend`：CPU NumPy 数组；
- `TorchInputBackend`：PyTorch Tensor；
- `PaddleInputBackend`：Paddle Tensor。

backend 之间不使用继承来复用行为。每个实现只负责 dtype、shape、随机值和基础数组操作；
API 语义不应写进 backend。

### `backend_runtime.py`：backend 运行策略和生命周期

这里集中处理：

- 环境变量、运行模式和显式参数的 backend 选择；
- backend factory；
- 设备策略、预热和进程级缓存；
- output grad 的独立随机流和 NumPy 缓存。

规则和调用方只消费解析后的 `InputBackendPolicy`，不应在各处重复读取环境变量。

### `value_generators.py`：通用值域生成

这里仅处理与 API 名称无关的通用逻辑，例如：

- 整数、浮点、复数和布尔值域；
- dtype 归一化；
- shape 相关的随机值；
- 通用索引、label 和 range 生成。

它只接收 `InputTensorSpec`、backend 和随机流，不读取 API 名称，也不修改 `TensorConfig`。

### `generation_rules.py`：所有 API 规则的唯一入口

这个文件故意保持为单文件，便于按 API 名称快速定位实现。它包含：

- 规则注册表和 decorator；
- `InputRuleContext`；
- 所有 API 专属的 shape、dtype、index、label 和参数关系规则；
- 未注册 API 的默认规则；
- `not_zero_apis` 等 API 语义策略。

规则只表达“这个 API 需要什么输入”。通用数值生成放到 `value_generators.py`，框架物化放到
`tensor_config.py` 和 `materialization.py`。

### `generation_rules.py`：选择并执行规则

`InputRuleRegistry` 根据 API 名称取得注册规则，创建上下文并执行一次生成事务。规则选择、上下文构造和提交生命周期共享同一入口。

## 规则怎么写

所有规则函数统一接收一个 `InputRuleContext`，不再维护 tuple 和 mapping 两套调用方式：

```python
@register("paddle._C_ops.example")
def generate_example(rule):
    x = rule.tensor("x")
    rule.set(x, rule.domain("random_range", x, 0, 1))
```

常用操作：

- `rule.arg(name, default)`：读取签名参数；
- `rule.tensor(name)`：要求参数名只对应一个 Tensor；
- `rule.tensors(name)`：取得参数名对应的全部 Tensor；
- `rule.domain()` / `rule.default()`：生成通用值域；
- `rule.generate(mapping)`：按参数名批量生成；
- `rule.set()`：写入逻辑值；
- `rule.value()`：读取已经写入的逻辑值；
- `rule.ops`：执行当前 backend 的基础数组操作。

规则执行结束时会统一检查：每个必需 Tensor 是否都有值、是否重复写入、是否需要保留原始规格。
检查失败会直接抛出异常，并放弃本次事务，不挂载部分结果。

## 必须保持的行为

- 相同 seed 和相同配置指纹应得到相同的逻辑输入值；
- NumPy、Torch、Paddle 使用各自的随机流，不能互相污染；
- backend 的最终设备由 `InputBackendPolicy` 决定；
- `TensorConfig` 的复制和共享语义不能改变；
- 规则失败时不写回部分输入；
- 未注册 API 仍使用默认规则；
- `generation_rules.py` 仍是 API 规则的唯一定位入口。

## 扩大默认浮点输入范围

设置 `PADDLEAPITEST_INPUT_MAX_ABS` 可调整普通默认浮点和复数输入的对称绝对上界：

```bash
PADDLEAPITEST_INPUT_MAX_ABS=10 python engineV4.py ...
```

此时默认范围为 `[-10, 10)`；复数的实部和虚部分别使用该范围。backward output-grad
使用独立的 `PADDLEAPITEST_OUTPUT_GRAD_MAX_ABS`，未设置时保持历史 `[-0.5, 0.5)`。
整数以及概率、索引、shape 等 API 专用值域不受影响。
该值必须是有限正数，并在运行时配置创建时冻结。

## 验证

代码变更后至少运行：

```bash
python -m compileall tester/input_generation tester
pytest tester/input_generation -q
pre-commit run check-added-comment-ratio
```

需要实际运行 API 时，优先用 `engineV4.py --api_config=...` 做单条配置验证，再根据环境运行批量回归。
