# 输入生成运行时

本目录只保留输入生成运行时代码，不再承载治理脚本或冻结报告。
`generation_rules.py` 保留为输入规则注册中心，`README.md` 同时承担模块地图和设计说明。

## 运行链路

```text
APIConfig -> APITestBase.generate_input_values -> dispatcher.py -> binding.py -> generation_rules.py -> value.py -> tensor_config.py
```

## 模块地图

- `tensor_path.py`：输入 Tensor 路径 `InputTensorPath`
- `tensor_spec.py`：张量只读描述 `InputTensorSpec`
- `binding.py`：签名解析、参数绑定、调用绑定和运行时上下文
- `value.py`：输入数据 `InputValue` 的读写、挂载和清理
- `value_generators.py`：与 API 无关的通用值生成，具体数组/tensor 由 backend 决定
- `generation_rules.py`：`@input_rules.register` 规则、`InputRuleContext` 和规则执行入口
- `dispatcher.py`：输入生成调度和规则查找
- `backend.py`：输入生成 backend 选择和 numpy/torch/paddle 实现
- `tensor_config.py`：张量配置、缓存和框架物化

规则接口优化计划见
`docs/superpowers/plans/2026-08-03-input-generation-rule-api-optimization.md`。

## 命名约束

- `input_generation` 包已经提供领域上下文，包内模块使用职责名，不重复添加 `input_` 前缀。
- 跨模块使用的类型、函数和全局对象继续使用 `Input` 或 `input_` 标识，离开包上下文后仍能识别归属。
- 规则模块使用 `generation_rules.py`，既表达输入构造职责，也避免与 `paddle_to_torch/rules.py` 重名。
- `Generation` 和 `Rule` 不单独作为顶层领域名称，分别使用 `InputGeneration` 和 `InputRule`。
- API 级规则统一命名为 `generate_<对象>_inputs`，例如 `generate_clip_inputs`。
- 单 Tensor 值生成器统一命名为 `generate_<值域>_input_value`。
- 流程变量明确区分 `api_config`、`input_binding`、`rule`、`input_value`、
  `input_backend` 和 `input_random_state`，不使用脱离上下文的 `config`、`binding` 或 `data`。
- backend 的 `reshape`、`uniform` 等标准数组原语，以及 `InputRuleContext` 的规则 DSL 短方法保留
  领域惯用名称，避免增加不必要的调用噪声。
- 单值生成函数内部允许使用数值计算惯例中的 `spec` 和 `rng`，函数名与类型注解负责标识输入领域。
- `tensor_config.py` 和 `TensorConfig` 保留现名；它们在全局没有歧义，也是配置 DSL 的稳定入口。

## 设计目标

- 未注册 API 使用注册表持有的独立默认规则
- 需要参数关系或特殊取值域的 API 使用显式注册规则，生成不完整时 fail fast
- 同一 seed 下保持 dtype、shape、bytes 和最终 RNG 状态一致
- 输入数据与框架输入构造分离，便于单独验证
- config 级 RNG 只在规则成功后提交，失败不污染后续 config

## 核心模型

### `InputTensorPath`

`InputTensorPath` 描述一次 API 调用中某个 Tensor 的稳定位置，例如 `args[0]`、`kwargs.x`、
`args[1][2]`。它替代 `index/key/list_index` 组合状态，避免规则依赖遍历顺序。

### `InputTensorSpec`

`InputTensorSpec` 是从 `TensorConfig` 提取出的只读快照，只保留值生成所需的 shape、dtype、
place、连续性和 strides。

### `InputApiBinding`

`InputApiBinding` 是规则侧看到的“一次调用”的绑定结果，包含 API 名称、已绑定参数、
Tensor 绑定和未解析原因。规则只应该读取它，不应该自己再做签名推断。

### `InputGenerationContext`

`InputGenerationContext` 封装绑定结果、配置指纹和 seed。NumPy RNG 使用独立状态副本，Torch/Paddle
backend 使用 seed 与配置指纹初始化自己的 generator。

### `InputValue`

`InputValue` 是输入数据容器。它是规则输出的真源；`TensorConfig` 只保留
`input_value` / `input_value_backend` 作为当前对象的逻辑值缓存，不再提供
`numpy_tensor` 兼容存储。

### `InputRule`

`InputRule` 表示一条 decorator 注册规则，负责执行规则函数、检查完整性并提交输入数据。

### 规则参数

所有规则函数只接收一个 `InputRuleContext`。它直接负责只读参数查询、Tensor 绑定和值域生成，并通过
私有 `_InputValueWriter` 暂存写入。规则使用 `rule.arg()`、`rule.tensor()`、`rule.tensors()`、
`rule.default()`、`rule.set()` 和 `rule.generate()`，不直接接触 writer 或 backend 实现。

同一个 `TensorConfig` 对象不能复用于多个 `InputTensorPath`。绑定阶段会直接拒绝这种配置，
错误信息同时给出首次路径和冲突路径，避免 path 寻址与对象 identity 寻址产生歧义。

### Backend Selection

未显式设置 backend 时 `use_gpu_mode=True` 自动使用 torch backend；
显式设置环境变量时按请求选择：

```bash
PADDLEAPITEST_INPUT_BACKEND=numpy
PADDLEAPITEST_INPUT_BACKEND=torch
PADDLEAPITEST_INPUT_BACKEND=paddle
```

未设置时默认 `numpy`。`torch` backend 的目标契约是 backend-native value：CPU 模式生成 CPU
`torch.Tensor`，GPU 模式生成 CUDA `torch.Tensor`，不转回 NumPy，也不保存 NumPy 数据。
`paddle` backend 同样保存 backend-native `paddle.Tensor`；accuracy 模式下 Torch 输入由
Paddle logical value 经 DLPack 生成，并在 Torch 侧 clone 为 Torch 自有存储。

`USE_CACHED_NUMPY=True` 时固定使用 `numpy` backend。如果同时设置
`PADDLEAPITEST_INPUT_BACKEND=torch` 或 `paddle`，框架会打印 warning 并忽略该 backend 请求。

## 规则编写方法

`generation_rules.py` 是唯一的规则入口。编写规则时遵循下面的顺序。

### 1. 先注册，再写逻辑

```python
@input_rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def generate_clip_inputs(rule: InputRuleContext):
    rule.generate(
        {
            "min": lambda tensor: rule.domain("random_range", tensor, -1, 0),
            "max": lambda tensor: rule.domain("random_range", tensor, 0, 1),
        }
    )
```

规则函数只描述 API 语义，不写绑定、物化或日志逻辑。

### 2. 优先使用 InputRuleContext

- `rule.arg(name, default)`：按签名参数名读取值
- `rule.tensor(name)`：要求参数名至多对应一个 Tensor；多个匹配会直接报错
- `rule.tensors(name)`：返回参数名对应的全部 Tensor 绑定
- `rule.ops`：执行 backend-native 数组或 Tensor 操作
- `rule.domain()` / `rule.default()`：生成值域数据
- `rule.generate(mapping)`：按参数名生成，未指定 Tensor 默认使用 default domain
- `rule.set()` / `rule.value()`：写入或读取逻辑值

### 3. 只在必要时写低层值

- `rule.set()`：正常写入并在成功完成时同步元数据
- `rule.set_preserving_spec()`：写入逻辑值，但保留配置原有 shape/dtype
- rule 结束后统一校验所有 Tensor 均已生成，再更新配置元数据并挂载数据

writer 在 rule 执行期间只暂存 backend 拷贝。rule 抛出异常或完整性检查发现遗漏时，
不会挂载部分 `InputValue`、修改 `TensorConfig` 元数据或提交 config RNG。重复写入同一 path
也会直接失败；rule 不应依赖覆盖已有值。

### 4. 失败要显式

不支持的参数关系应直接抛出带 API 上下文的异常，不要让默认分支悄悄补值。注册表不保留没有
实际消费者的 GPU/cache 阻断元数据。

### 5. 值域逻辑和 API 逻辑分离

`value_generators.py` 只负责 dtype、shape 和 value domain，不读取 API 名称。API 专属关系应留在
`generation_rules.py` 的输入规则函数里。

## 与旧实现的区别

### 1. 从单体条件链改为显式模块

最初的 `config_analyzer` 把解析、绑定、值生成、物化和调度都塞进一个大类里，再用长
`if/elif api_name` 条件链分派。现在这些职责拆成 `binding.py`、`generation_rules.py`、
`dispatcher.py`、`value_generators.py`、`value.py` 和 `tensor_config.py`，边界更清楚。

### 2. 从逐参数副作用改为完整 config 规则

旧实现常在生成一个参数时顺手读写另一个参数，行为依赖遍历顺序。现在规则以完整 config 为单位
描述，并统一通过 `InputRuleContext` 显式表达参数关系。

### 3. 从生成逻辑与物化耦合改为分层处理

过去“生成什么值”和“如何创建 Paddle/Torch 张量”是绑在一起的。现在规则先产出输入数据，
再由 `tensor_config.py` 负责框架张量构造。这样可以单独校验 dtype、shape、bytes 和 RNG 状态。

### 4. 从隐藏 fallback 改为显式默认规则和 fail-fast

纯输入生成且不依赖参数关系的 API 走默认规则；需要特殊语义的 API 必须显式注册。
已注册规则重复写入或遗漏 Tensor 时直接失败，避免静默产出错误输入。

### 5. 从全局随机状态改为 config-owned RNG

旧实现更接近共享全局随机状态。现在输入生成持有 config 级 RNG，规则成功后才提交状态，
失败不会污染其他 config。

## 运行时约束

- `InputConfigRandomState` 使用独立 `RandomState` 副本
- 同一 `TensorConfig` 复用于多个输入 path 时，`binding.py` 直接拒绝
- `dispatcher.py` 只做规则查找和上下文构造
- `value.py` 只管理输入数据，不承担框架创建
- `tensor_config.py` 只承担 Tensor 物化和读写

## 当前状态

- `InputConfigRandomState` 使用独立 `RandomState` 副本，并在规则成功后提交到全局
- 所有注册规则统一使用单参数 `InputRuleContext` 协议
- 运行时目录只保留当前实现代码
