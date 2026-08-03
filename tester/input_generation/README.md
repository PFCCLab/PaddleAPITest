# 输入生成运行时

本目录只保留输入生成运行时代码，不再承载治理脚本或冻结报告。
`input_registry.py` 保留为输入规则注册中心，`README.md` 同时承担模块地图和设计说明。

## 运行链路

```text
APIConfig -> APITestBase.gen_input_data -> input_dispatch.py -> input_binding.py -> input_registry.py -> input_data.py -> tensor_config.py
```

## 模块地图

- `tensor_path.py`：输入 Tensor 路径 `TensorPath`
- `tensor_spec.py`：张量只读描述 `TensorSpec`
- `input_binding.py`：签名解析、参数绑定、调用绑定和运行时上下文
- `input_data.py`：输入数据 `InputData` 的读写、挂载和清理
- `value_gen.py`：与 API 无关的通用值生成，具体数组/tensor 由 backend 决定
- `input_registry.py`：`@rules.register` 规则、`ConfigView` / `ValueFactory` / `InputWriter` 和规则执行入口
- `input_dispatch.py`：输入生成调度、规则查找和阻断
- `input_backend.py`：输入生成 backend 选择和 numpy/torch/paddle 实现
- `tensor_config.py`：张量配置、缓存和框架物化

输入生成后端抽象和 backend-native 迁移的设计与进度见：

- `docs/superpowers/specs/2026-08-02-backend-native-input-generation-design.md`
- `docs/superpowers/plans/2026-08-02-backend-native-input-generation-migration.md`

## 设计目标

- 默认路径先进入规则注册表，未注册 API 使用默认输入生成规则
- 需要参数关系或特殊取值域的 API 使用显式注册规则，规则被阻断或生成不完整时 fail fast
- 同一 seed 下保持 dtype、shape、bytes 和最终 RNG 状态一致
- GPU 或缓存未准备好时，在生成前直接阻断
- 输入数据与框架输入构造分离，便于单独验证
- config 级 RNG 只在规则成功后提交，失败不污染后续 config

## 核心模型

### `TensorPath`

`TensorPath` 描述一次 API 调用中某个 Tensor 的稳定位置，例如 `args[0]`、`kwargs.x`、
`args[1][2]`。它替代 `index/key/list_index` 组合状态，避免规则依赖遍历顺序。

### `TensorSpec`

`TensorSpec` 是从 `TensorConfig` 提取出的只读快照，只保留值生成所需的 shape、dtype、
place、连续性和 strides。

### `InputBinding`

`InputBinding` 是规则侧看到的“一次调用”的绑定结果，包含 API 名称、参数绑定、
Tensor 绑定和未解析原因。规则只应该读取它，不应该自己再做签名推断。

### `InputContext`

`InputContext` 把一次生成需要的上下文一次性封装起来：绑定结果、配置指纹、seed
和 GPU 开关。规则不应直接依赖全局状态。

### `InputData`

`InputData` 是输入数据容器。它是规则输出的真源；`TensorConfig` 只保留
`input_value` / `input_value_backend` 作为当前对象的逻辑值缓存，不再提供
`numpy_tensor` 兼容存储。

### `RegisteredRule`

`RegisteredRule` 表示一条 decorator 注册规则，负责校验 GPU/cache 阻断、执行规则函数、
检查完整性并提交输入数据。

### 规则参数

规则函数接收三个职责明确的参数：

- `ConfigView`：只读 config 视图，负责原始参数读取、Tensor 绑定查询和 API 名称
- `ValueFactory`：数值生成入口，负责 value domain 和 backend-native 数组/tensor 操作
- `InputWriter`：输入写入入口，负责暂存、去重、读取已生成值和完成校验

常见批量生成流程由 `generate_all()`、`generate_remaining()` 和
`generate_by_parameter()` 承担。按单个参数处理时，先通过 `config.binding()` 或
`config.bindings()` 得到绑定，再显式调用 writer。

同一个 `TensorConfig` 对象不能复用于多个 `TensorPath`。绑定阶段会直接拒绝这种配置，
错误信息同时给出首次路径和冲突路径，避免 path 寻址与对象 identity 寻址产生歧义。

### Backend Selection

当前 backend 迁移目标见：

- `docs/superpowers/specs/2026-08-02-backend-native-input-generation-design.md`
- `docs/superpowers/plans/2026-08-02-backend-native-input-generation-migration.md`

目标状态下，未显式设置 backend 时 `use_gpu_mode=True` 自动使用 torch backend；
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

`input_registry.py` 是唯一的规则入口。编写规则时遵循下面的顺序。

### 1. 先注册，再写逻辑

```python
@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", "default"),
            ("min", lambda binding: values.domain("random_range", binding, -1, 0)),
            ("max", lambda binding: values.domain("random_range", binding, 0, 1)),
        ),
    )
```

规则函数只描述 API 语义，不写绑定、物化或日志逻辑。

### 2. 优先使用 registry helper

- `generate_all(config, values, writer, "default")`：同类参数统一生成
- `generate_remaining(config, values, writer, "default")`：补齐未生成参数
- `generate_by_parameter(config, values, writer, ...)`：按参数名映射不同生成策略
- `config.arg()` / `config.kwarg()`：读取原始参数
- `config.binding()`：要求参数名至多对应一个 Tensor；多个匹配会直接报错
- `config.bindings()`：返回参数名对应的全部 Tensor 绑定
- `config.binding_for_value()`：按 `TensorConfig` 对象定位绑定
- `writer.value()`：读取逻辑值

### 3. 只在必要时写低层值

- `writer.set_value()`：正常写入并在成功完成时同步元数据
- `writer.set_value_preserving_spec()`：写入逻辑值，但保留配置原有 shape/dtype
- `writer.finish(config)`：校验所有 Tensor 均已生成，再统一更新配置元数据并返回待挂载数据

writer 在 rule 执行期间只暂存 backend 拷贝。rule 抛出异常或 `finish()` 发现遗漏时，
不会挂载部分 `InputData`、修改 `TensorConfig` 元数据或提交 config RNG。重复写入同一 path
也会直接失败；rule 不应依赖覆盖已有值。

### 4. 失败要显式

如果某个规则不支持 GPU、缓存或某种参数关系，直接返回阻断原因，不要让默认分支悄悄补值。
`block_reason()` 的职责就是在生成前失败。

### 5. 值域逻辑和 API 逻辑分离

`value_gen.py` 只负责 dtype、shape 和 value domain，不读取 API 名称。API 专属关系应留在
`input_registry.py` 的 rule 函数里。

## 与旧实现的区别

### 1. 从单体条件链改为显式模块

最初的 `config_analyzer` 把解析、绑定、值生成、物化和调度都塞进一个大类里，再用长
`if/elif api_name` 条件链分派。现在这些职责拆成 `input_binding.py`、`input_registry.py`、
`input_dispatch.py`、`value_gen.py`、`input_data.py` 和 `tensor_config.py`，边界更清楚。

### 2. 从逐参数副作用改为完整 config 规则

旧实现常在生成一个参数时顺手读写另一个参数，行为依赖遍历顺序。现在规则以完整 config 为单位
描述，通过 `generate_by_parameter()`、`generate_all()`、`config.arg()` 和 `config.kwarg()`
显式表达参数关系。

### 3. 从生成逻辑与物化耦合改为分层处理

过去“生成什么值”和“如何创建 Paddle/Torch 张量”是绑在一起的。现在规则先产出输入数据，
再由 `tensor_config.py` 负责框架张量构造。这样可以单独校验 dtype、shape、bytes 和 RNG 状态。

### 4. 从隐藏 fallback 改为显式默认规则和 fail-fast

纯输入生成且不依赖参数关系的 API 走默认规则；需要特殊语义的 API 必须显式注册。
已注册规则被阻断、重复写入或遗漏 Tensor 时直接失败，避免静默产出错误输入。

### 5. 从全局随机状态改为 config-owned RNG

旧实现更接近共享全局随机状态。现在输入生成持有 config 级 RNG，规则成功后才提交状态，
失败不会污染其他 config。

## 运行时约束

- `ConfigNumpyRNG` 使用独立 `RandomState` 副本
- 同一 `TensorConfig` 复用于多个输入 path 时，`input_binding.py` 直接拒绝
- `input_dispatch.py` 只做规则查找和阻断
- `input_data.py` 只管理输入数据，不承担框架创建
- `tensor_config.py` 只承担 Tensor 物化和读写

## 当前状态

- `ConfigNumpyRNG` 使用独立 `RandomState` 副本，并在规则成功后提交到全局
- GPU 和 cached 阻断通过 `allow_gpu` / `allow_cached` 控制
- 运行时目录只保留当前实现代码
