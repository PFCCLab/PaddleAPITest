# 输入生成运行时

本目录只保留输入生成运行时代码，不再承载治理脚本或冻结报告。
`registry.py` 保留为规则注册中心，`README.md` 同时承担模块地图和设计说明。

## 运行链路

```text
APIConfig -> APITestBase.gen_numpy_input -> dispatch.py -> input_bind.py -> registry.py -> input_values.py -> tensor_config.py
```

## 模块地图

- `input_path.py`：输入路径 `InputPath`
- `tensor_view.py`：张量只读视图 `TensorView`
- `input_bind.py`：签名解析、参数绑定、调用绑定和运行时上下文
- `input_values.py`：逻辑值 `InputValue` 的读写、挂载和清理
- `value_sample.py`：与 API 无关的纯 NumPy 值生成
- `registry.py`：`@rules.register` 规则、`RuleCase` 和规则执行入口
- `dispatch.py`：输入生成调度、规则查找和阻断
- `tensor_config.py`：张量配置、缓存和框架物化

## 设计目标

- 默认路径直接进入规则注册表
- 只接受显式 allowlist 规则，未注册 API 直接 fail fast
- 同一 seed 下保持 dtype、shape、bytes 和最终 RNG 状态一致
- GPU 或缓存未准备好时，在生成前直接阻断
- 逻辑值与框架物化分离，便于单独验证
- case 级 RNG 只在规则成功后提交，失败不污染后续 case

## 核心模型

### `InputPath`

`InputPath` 描述一次 API 调用中某个 Tensor 的稳定位置，例如 `args[0]`、`kwargs.x`、
`args[1][2]`。它替代 `index/key/list_index` 组合状态，避免规则依赖遍历顺序。

### `TensorView`

`TensorView` 是从 `TensorConfig` 提取出的只读快照，只保留值生成所需的 shape、dtype、
place、连续性和 strides。

### `BoundInput`

`BoundInput` 是规则侧看到的“一次调用”的绑定结果，包含 API 名称、参数绑定、
Tensor 绑定和未解析原因。规则只应该读取它，不应该自己再做签名推断。

### `InputGenerationContext`

`InputGenerationContext` 把一次生成需要的上下文一次性封装起来：绑定结果、配置指纹、seed、
Torch 开关和 GPU 开关。规则不应直接依赖全局状态。

### `InputValue`

`InputValue` 是逻辑值容器。它是规则输出的真源，`TensorConfig.numpy_tensor` 只作为备用存储。

### `RegisteredRule`

`RegisteredRule` 表示一条 decorator 注册规则，负责校验 GPU/cache 阻断、执行规则函数、
检查完整性并提交 payload。

### `RuleCase`

`RuleCase` 是规则编写时拿到的可变 case 视图。它负责：

- 按 `InputPath` 读取和写入 Tensor
- 维护 case 级 RNG
- 提供 `generate()`、`generate_all()`、`generate_remaining()`、`generate_by_parameter()`
- 防止重复写入
- 收集最终 payload

## 规则编写方法

`registry.py` 是唯一的规则入口。编写规则时遵循下面的顺序。

### 1. 先注册，再写逻辑

```python
@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate("x", "default")
    case.generate("min", "random_range", low=-1, high=0)
    case.generate("max", "random_range", low=0, high=1)
```

规则函数只描述 API 语义，不写绑定、物化或日志逻辑。

### 2. 优先使用 `case` 的高层接口

- `case.generate_all("default")`：同类参数统一生成
- `case.generate("x", "default")`：针对参数名生成
- `case.generate_remaining("default")`：补齐未生成参数
- `case.generate_by_parameter(...)`：按参数名映射不同生成策略
- `case.arg()` / `case.kwarg()`：读取原始参数
- `case.find()`：按参数名定位绑定
- `case.value()`：读取逻辑值

### 3. 只在必要时写低层值

- `case.set_value()`：正常写入并同步元数据
- `case.rewrite_value()`：重写已有值
- `case.set_value_raw()`：仅在需要保留元数据时使用

### 4. 失败要显式

如果某个规则不支持 GPU、缓存或某种参数关系，直接返回阻断原因，不要让默认分支悄悄补值。
`block_reason()` 的职责就是在生成前失败。

### 5. 值域逻辑和 API 逻辑分离

`value_sample.py` 只负责 dtype、shape 和 value domain，不读取 API 名称。API 专属关系应留在
`registry.py` 的 rule 函数里。

## 与旧实现的区别

### 1. 从单体条件链改为显式模块

最初的 `config_analyzer` 把解析、绑定、值生成、物化和调度都塞进一个大类里，再用长
`if/elif api_name` 条件链分派。现在这些职责拆成 `input_bind.py`、`registry.py`、
`dispatch.py`、`value_sample.py`、`input_values.py` 和 `tensor_config.py`，边界更清楚。

### 2. 从逐参数副作用改为完整 case 规则

旧实现常在生成一个参数时顺手读写另一个参数，行为依赖遍历顺序。现在规则以完整 case 为单位
描述，通过 `case.generate()`、`case.generate_all()`、`case.arg()` 和 `case.kwarg()`
显式表达参数关系。

### 3. 从生成逻辑与物化耦合改为分层处理

过去“生成什么值”和“如何创建 Paddle/Torch 张量”是绑在一起的。现在规则先产出逻辑 payload，
再由 `tensor_config.py` 负责框架张量构造。这样可以单独校验 dtype、shape、bytes 和 RNG 状态。

### 4. 从隐式 fallback 改为显式 allowlist 和 fail-fast

`config_analyzer` 时代的默认分支容易掩盖遗漏特例。现在只接受显式注册的 API allowlist，
未注册或被阻断的规则都会直接失败，避免静默回退。

### 5. 从全局随机状态改为 case-owned RNG

旧实现更接近共享全局随机状态。现在 `RuleCase` 持有 case 级 RNG，规则成功后才提交状态，
失败不会污染其他 case。

## 运行时约束

- `CaseNumpyRNG` 使用独立 `RandomState` 副本
- `dispatch.py` 只做规则查找和阻断
- `input_values.py` 只管理逻辑值，不承担框架创建
- `tensor_config.py` 只承担 Tensor 物化和读写

## 当前状态

- `CaseNumpyRNG` 使用独立 `RandomState` 副本，并在规则成功后提交到全局
- GPU 和 cached 阻断通过 `allow_gpu` / `allow_cached` 控制
- 运行时目录只保留当前实现代码
