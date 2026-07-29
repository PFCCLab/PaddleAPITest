# 测试输入生成治理设计

## 1. 文档目的

本文解释 PaddleAPITest 输入生成重构的目标、当前实现、迁移策略和长期形态。
重点回答以下问题：

1. 为什么不能直接继续维护旧的 `api_name` 条件链。
2. 为什么当前需要 dispatcher、registry、binding、model、value generator 等模块。
3. 为什么新实现必须与 legacy 双轨运行，而不是一次性替换。
4. 当前哪些代码是长期架构，哪些只是迁移期兼容层。
5. 一个 API 应如何迁移，以及如何证明迁移没有改变生成语义。

本文描述的是输入生成子系统，不改变配置文件文本协议、Paddle/Torch 对齐逻辑或
测试执行框架。

## 2. 原实现的问题

旧实现将以下职责集中在 `config_analyzer.py` 的一个大型类中：

- 配置文本解析。
- 参数身份判断。
- 每个 API 的输入约束。
- NumPy 随机值生成。
- Paddle/Torch 张量物化。
- GPU 共享数据处理。
- 缓存与对象生命周期。

其中输入生成采用“遍历一个参数，调用一次 `get_numpy_tensor()`”的方式，并在
函数内部通过大型 `if/elif api_config.api_name` 条件链决定行为。这带来四类核心
问题。

### 2.1 参数身份不稳定

生成逻辑依赖 `self.index`、`self.key` 和 `self.list_index` 判断当前参数。位置参数、
关键字参数和嵌套 Tensor 列表使用不同的隐式状态，规则很难直接表达“正在生成
哪个参数”。

### 2.2 完整 case 被拆成逐参数副作用

很多 API 的输入之间存在关系，例如：

- `min <= max`。
- index 必须落在输入维度范围内。
- label 必须小于类别数。
- reshape 前后 numel 必须一致。
- linalg 输入需要满足正定、三角或满秩约束。

旧实现生成当前参数时会读取或直接写入其他 `TensorConfig.numpy_tensor`。因此结果
依赖 args/kwargs 遍历顺序，很难单独测试一条规则，也无法安全并行或重排参数。

### 2.3 逻辑值与框架物化耦合

“生成什么值”和“如何创建 Paddle/Torch/GPU Tensor”混在一起。修改随机值域可能
意外影响 DLPack、非连续 Tensor、BF16/FP8 中间 dtype 或缓存行为。

### 2.4 缺少可证明的迁移边界

条件链末尾存在默认生成逻辑，但不能把它注册成所有 API 的 catch-all。任何遗漏的
API 特例、`startswith` family 或动态集合规则都会被 catch-all 静默绕过，测试仍能
运行，却已经改变输入语义。

## 3. 设计目标

### 3.1 当前迁移阶段目标

- 默认执行路径保持 legacy，不影响现有任务。
- v2 只接管经过验证的显式 API allowlist。
- 同一 seed 下保持 NumPy 数组 dtype、shape、bytes 和最终 RNG 状态一致。
- GPU 或 cache 尚未迁移时，在调用新 rule/value generator 前回退 legacy。
- 未迁移 API 继续调用原来的完整 case 生成循环。
- 将规则命中、回退原因和 legacy 使用情况变为可观测信息。

### 3.2 长期目标

长期数据流如下：

```text
配置文本
  -> APIConfig
  -> BoundCall + ArgPath
  -> GenerationContext
  -> @rules.register API case rule
  -> RuleCase 生成逻辑 payload
  -> TensorConfig.numpy_tensor
  -> Paddle/Torch/GPU Materializer
```

最终应满足：

- 每个已迁移 API 的初始化语义由一个直观的 decorator rule 表达。
- 参数通过稳定路径标识，不依赖遍历产生的可变状态。
- value generator 只描述 dtype/shape/value domain，不读取 API 名称。
- rule 显式描述 API 和参数关系。
- Paddle 与 Torch 从同一个逻辑 payload 物化。
- RNG 和 cache 具有明确的 case/session 所有权。
- legacy fallback 归零后删除旧条件链。

### 3.3 当前明确不做的事情

- 不调整随机分布或数值范围。
- 不切换 NumPy RNG 算法。
- 不一次性迁移跨参数、optimizer、MoE、FP8 或后端相关规则。
- 不用 YAML 表达控制流和复杂参数关系。
- 不因为代码看起来重复就提前抽象尚未稳定的领域规则。

## 4. 核心模型

### 4.1 `ArgPath`

`ArgPath` 是 Tensor 参数在一次 API 调用中的稳定身份，例如：

```text
args[0]
kwargs.x
args[1][2]
kwargs.inputs[0]
```

它替代 `index/key/list_index` 的组合判断，使规则、payload、日志和验证工具使用同一种
参数定位方式。

### 4.2 `TensorSpec`

`TensorSpec` 是只读的逻辑 Tensor 描述，当前包含：

- shape
- dtype
- place
- contiguous 属性
- strides

value generator 接收 `TensorSpec`，不直接读取或修改 `TensorConfig`。

### 4.3 `BoundCall`

`BoundCall` 将配置中的 args/kwargs 绑定到 API 参数名，并收集所有 Tensor 的
`ArgPath + TensorSpec`。它解决“位置 1 到底是 `y`、`axis` 还是其他参数”的问题。

绑定层当前仍以 shadow 和 v2 命中路径为主。legacy 默认和未注册 API fallback 不会
加载或执行 binding，避免新增副作用。

### 4.4 `GenerationContext`

`GenerationContext` 是规则执行所需的只读上下文，包含：

- `BoundCall`
- 配置 fingerprint
- seed
- runtime mode
- Paddle/Torch 模式
- GPU 能力

它不保存框架 Tensor，也不作为全局可变状态使用。

### 4.5 `RuleCase`

`RuleCase` 是 decorator rule 面向 API 作者的生成接口。规则不直接遍历
`TensorConfig`，也不再声明一行“参数名到 generator”的静态数据，而是显式写出当前 API
的生成顺序：

```python
@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(ctx, case):
    case.generate("x", "default")
    min_value = case.generate("min", bounded(...))
    case.generate("max", greater_than(min_value))
```

当前实现已经使用 `case.generate()`、`case.generate_all()` 和
`case.generate_remaining()` 生成逻辑 payload。`RegisteredRule` 会把 payload 挂到
`api_config`，Paddle/Torch materializer 优先从 payload 取逻辑值。当前仍同步回写
`TensorConfig.numpy_tensor` 作为回退桥；rule 的写法不应随这个内部所有权变化而大改。

少数 legacy 分支会只写 `TensorConfig.numpy_tensor`，但故意不更新 `shape`/`dtype`
metadata。当前 `RuleCase.set_value_raw()` 只作为迁移期兼容桥接使用，用来复刻这类可观测
行为；普通 rule 仍应优先使用 `case.generate()`、`case.set_value()` 或
`case.generate_by_parameter()`。

像 `max_unpool1d/2d/3d` 这类需要先借助 Paddle 后端生成合法 `x/indices` 的分支，
应写成专用 decorator rule，并把后端预生成逻辑限制在 rule-local helper 内，而不是继续保留
为 deferred case。

`RuleCase` 必须保证：

- 每个 Tensor path 最多生成一次。
- rule 结束时所有 Tensor path 都已生成，或在任何 RNG 调用前 fallback。
- 生成顺序与 legacy 可对照。
- 跨参数关系通过 `RuleCase` 的返回值、引用或 validator 表达，而不是读写其他
  `TensorConfig.numpy_tensor`。
- rule 作者不直接调用 `numpy`、`numpy.random` 或 `LEGACY_NUMPY_RNG`。数组构造和随机数
  通过 `case.array()`、`case.zeros()`、`case.ones()`、`case.random()`、
  `case.randint()`、`case.choice()`、`case.value_domain()` 等包装接口完成；当前这些接口
  通过 `CaseNumpyRNG(backend="legacy-global")` 代理全局 NumPy RNG，并记录 case 的
  seed/fingerprint/runtime metadata，后续再切换为真正独立的 case-owned RNG state。
- rule 作者通过 `case.arg(position, name, default)`、`case.kwarg(name, default)`、
  `case.has_kwarg(name)` 和 `case.is_tensor_config(value)` 读取 API 配置形态；不要在规则体中
  直接访问 `raw_case.args/kwargs` 或调用 `_tensor_config_at()`。

## 5. Rule 与 Value Generator 的区别

### 5.1 Value Generator

Value generator 回答“给定一个 `TensorSpec`，应该生成怎样的逻辑 NumPy 值”。它不能读取
`api_name`，也不能访问其他参数。

当前已有：

- `generate_default`
- `generate_nonzero`
- `generate_unit_interval`
- `generate_uniform`
- `generate_multiply`
- 以及少量保留 legacy 细节的共用值域函数，例如 `generate_full_fill_value`、
  `generate_dropout_probability`、`generate_remainder_rhs`

例如 nonzero value generator 保留了 legacy 的精确行为：

- int8 使用 `randint(1, 256, dtype=int32)`，再把 128 至 255 映射到负数。
- uint8 使用 `[1, 256)`。
- 其他整数使用 `[1, 65535)`。
- 浮点使用 `random + 0.5`。
- 复数分别生成实部和虚部，因此 RNG 调用次数也必须保持一致。

### 5.2 Decorator Rule 与 API_RULE_REGISTRY

当前 API 关联使用 `registry.py` 中的 decorator registry。每条规则用
`@rules.register(...)` 声明 API、alias 和可选 fallback，然后在函数体中写出
完整 case 的初始化流程：

```python
@rules.register("paddle.normal")
def normal_values(ctx, case):
    case.generate_by_parameter(
        (
            ("mean", "signed_half_interval"),
            ("std", "normal_std"),
        ),
        default="int_zero_1024",
    )
```

同一个值域可以被多行复用。例如 `paddle.bernoulli`、`paddle.standard_gamma` 和
`paddle.poisson` 都使用 `unit_interval`，但各自保留独立 decorator rule。这样读者可以
直接从 API rule 入口看到该 API 的初始化语义，而不需要在静态表、
参数映射和 generator 名称之间来回跳转。

`API_RULE_REGISTRY` 只是 decorator 注册完成后生成的 `api_name -> RegisteredRule` 索引，
用于 dispatcher 查找和冲突检测；它不是人工维护的规则表。规则真源是装饰器函数。

当前简单规则已经按 case-level decorator 执行，只是多数函数内部调用
`case.generate_all()` 或 `case.generate_remaining()`。后续 `clip`、reshape、index、
label/loss 等跨参数规则会继续使用同一个 rule 形态，只扩展 `RuleCase` 的关系表达能力。

规则编写规范是：

- API 专属逻辑优先写在对应的 `@rules.register(...)` 函数体内。
- 如果一段逻辑较长，可以使用 rule 函数内的局部 helper；不要把只服务单个 API 的行为提升
  为模块级 `_xxx_value()`，否则读者仍需要在 decorator 和外部函数之间跳转。
- rule-local helper 只接收当前 `binding`，例如 `def label_value(binding): ...`；需要
  API 参数、RNG、数组或已生成值时通过闭包里的 `case` 访问，不向 helper 继续传递
  `context/raw_case`。
- 可跨 API 复用、且只依赖 `TensorSpec/dtype/shape/value domain` 的逻辑，放入
  `value_generators.py`。
- decorator rule 体不直接使用 `numpy` 或 `LEGACY_NUMPY_RNG`；需要随机数、数组、choice、
  zeros/ones 等操作时使用 `RuleCase` 包装方法。
- `generate_by_parameter()` 只用于简单参数名到通用 value domain 的分派，或用于保持 legacy
  Tensor 遍历/RNG 顺序；不应把大量 API 独有逻辑隐藏到模块级 helper 后再交给它调用。
- 少数 legacy 分支会命中 API 但不写入某个 Tensor，例如
  `generate_proposals.variances`。这类配置必须在任何 RNG 调用前 fallback，不能擅自补
  default 值改变 legacy 可观测结果。

### 5.3 为什么必须分开

同一种值域可能服务多个 API，而同一个 API 也可能对不同参数使用不同值域。若
value generator 直接判断 API 名称，大型条件链只会换一个文件继续增长；若 rule 直接重复
所有 dtype 细节，则基础随机生成逻辑又会分散到各 API 中。

因此边界是：

```text
API allowlist、参数顺序、跨参数关系和 fallback 归 @rules.register rule
dtype/shape/value domain 归 value generator
API_RULE_REGISTRY 只是 decorator 产物，用于调度查找
```

### 5.4 长期最终形态

长期 rule 写法应接近以下形式：

```python
@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(ctx, case):
    x = case.generate("x", default())
    lower = case.generate("min", bounded_like(x))
    case.generate("max", greater_than(lower))
```

其中：

- `@rules.register(...)` 是唯一 API 规则注册方式，负责 API allowlist、alias、fallback hook
  和冲突检测。
- rule 函数是 API 初始化语义的唯一可读入口；简单 API 可以只写
  `case.generate_all(default())`，复杂 API 写出参数之间的关系。
- `default()`、`uniform(0, 1000)`、`greater_than("min")`、`axis_of("x")` 等是
  value-domain descriptor。当前实现用字符串 generator 名称过渡，后续可替换成 descriptor
  对象，但不应回到静态参数映射表。
- descriptor 最终由共用 value generator 执行 dtype、shape 和 RNG 细节；value generator
  不读取 API 名称。
- `RuleCase` 保存已生成 payload、按 `ArgPath` 去重、检查完整性，并在 rule 结束后运行
  validator。
- materializer 只消费 `RuleCase` 产出的逻辑 payload；Paddle/Torch/GPU 不再从
  `TensorConfig` 上读取可变中间状态。
- 需要借助框架后端先构造合法输入的分支（例如 `max_unpool*`）应改写为专用 decorator rule，
  后端预生成逻辑只作为 rule-local helper 存在。
- 共享 optimizer family（例如 `OPTIMIZER_APIS`）已经收口到专用 decorator rule，不再属于
  deferred 处理面。
- 不支持的配置必须在任何 RNG 调用前 fallback，例如 rule 显式设置
  `allow_gpu=False` / `allow_cached=False`，或 rule 暂不支持 `dropout.axis` 这类关系参数。

领域拆分后的长期目录可以是：

```text
input_generation/
  registry.py        # rules.register、RuleCase、RegisteredRule、索引构建
  value_generators.py
  rules/
    creation.py
    elementwise.py
    indexing.py
    loss.py
    linalg.py
  materializers/
    numpy_payload.py
    paddle.py
    torch.py
```

所有 `rules/*.py` 仍使用同一个 decorator registry。禁止同时存在“decorator rule”和
“静态 InputGenerationRule 表”两套生产规则系统。

## 6. 双轨调度

入口为 `APITestBase.gen_numpy_input()`，由 dispatcher 选择执行路径。

### 6.1 默认 legacy

```text
PADDLEAPITEST_INPUT_GENERATOR 未设置或为 legacy
  -> 记录 legacy dispatch
  -> 调用原完整 case 生成循环
```

这是当前生产默认值，因此注册新规则不会自动改变现有测试任务。

### 6.2 v2 命中规则

```text
PADDLEAPITEST_INPUT_GENERATOR=v2
  -> API_RULE_REGISTRY 按完整 api_name 查找 decorator rule
  -> 构建 BoundCall / GenerationContext
  -> 检查 rule 级 fallback 条件
  -> 执行 decorator rule
  -> 写入对应 TensorConfig
```

### 6.3 v2 未命中或不支持

```text
未注册 API
  -> fallback_reason = no-registered-rule
  -> 调用 legacy 完整 case 循环

已注册但当前 runtime mode 被 rule 显式禁用
  -> 在 rule 消耗 RNG 前记录 fallback
  -> 调用 legacy 完整 case 循环
```

回退单位必须是完整 case，不能先生成部分参数再调用 legacy，否则 RNG 顺序和参数间
关系都会改变。

## 7. 为什么使用显式 Allowlist

当前 `API_RULE_REGISTRY` 不提供 catch-all rule。每个迁移 API 都必须明确列出，原因如下：

1. legacy 默认分支只在所有前置特例均未命中时执行。
2. inventory 无法仅靠 API 字面量完整表示动态 family 和集合规则。
3. catch-all 会让遗漏特例表现为“成功使用 v2”，缺少明显失败信号。
4. 显式 allowlist 可以准确计算迁移数量和 legacy fallback 数量。
5. decorator 注册时检查 API 重叠，而不是依赖导入顺序覆盖。

新增默认 value generator 并不意味着可以把所有 API 注册到默认 rule。必须先证明该 API 在
所有目标配置语料中都只走 legacy 默认分支。

## 8. 当前迁移范围

当前 production registry 包含 106 个 decorator rule，显式覆盖 172 个 API。`rule_id`
不再是规则作者需要声明或维护的概念；telemetry 事件中的 `rule_id` 字段仅为兼容既有
schema 保留，值由 decorator 的主 API 自动派生。

| Decorator entry | Value domain | 显式范围 |
|---|---|---|
| `default_values` | default | `paddle.add`、`paddle.logical_not`、`paddle.concat` |
| `nonzero_values` | nonzero | legacy `not_zero_apis` 中的 18 个 API |
| `bernoulli_probability` | probability | `paddle.bernoulli` |
| `standard_gamma_unit_interval` | `[0, 1)` | `paddle.standard_gamma` |
| `poisson_unit_interval` | `[0, 1)` | `paddle.poisson` |
| `sqrt_nonnegative` | `[0, 1000)` | `paddle.sqrt`、`paddle.Tensor.sqrt` |
| `rsqrt_positive` | `[1e-7, 1000)` | `paddle.rsqrt`、`paddle.Tensor.rsqrt` |
| `multiply_values` | multiply legacy unit interval/complex | `paddle.multiply` |
| `binary_cross_entropy_values` | `[0, 1)` | `paddle.nn.functional.binary_cross_entropy` |
| `alpha_dropout_values` | `x` 为 `[0, 1)`，其他 Tensor default | `paddle.nn.functional.alpha_dropout` |
| `zero_65535_or_unit_values` | int `[0,65535)`，其他 `[0,1)` | `paddle.gammainc`、`paddle.gammaincc`、`paddle.linspace` |
| `dot_values` | int `[-127,127)`，其他 default | `paddle.dot` |
| `normal_values` | `mean/std/shape` 参数各自 legacy 值域 | `paddle.normal` |
| `ones_shape` | legacy shape integer range | `paddle.ones` |
| `zeros_shape` | legacy shape integer range | `paddle.zeros` |
| `eye_shape` | legacy shape integer range | `paddle.eye` |
| `shape_parameter_values` | `size/scale_factor/repeat_times` 为 `[1,128)`，其他 Tensor default | `paddle.nn.functional.interpolate`、`paddle.tile`、`paddle.Tensor.tile` |
| `upsample_values` | `size` 为 `[1,128)`，`scale_factor` 为 `[1,2)`，其他 Tensor default | `paddle.nn.functional.upsample` |
| `gaussian_nll_loss_values` | `var/variance` 为 `[1,2)`，其他 Tensor default | `paddle.nn.functional.gaussian_nll_loss` |
| `hinge_embedding_loss_values` | `label` 为 `{-1,1}`，其他 Tensor default | `paddle.nn.functional.hinge_embedding_loss` |
| `sigmoid_focal_loss_values` | `label` 为 `{0,1}`，其他 Tensor default | `paddle.nn.functional.sigmoid_focal_loss` |
| `full_values` | `shape` 为 `[0,64)`，`fill_value` 为 legacy 非零值域 | `paddle.full` |
| `standard_normal_shape` | `shape` 为 `[1,128)`，其他 Tensor default | `paddle.standard_normal` |
| `logspace_values` | `num` 为 `[1,65535)` 且保留 legacy no-cast dtype | `paddle.logspace` |
| `quantile_values` | `q` 为 legacy `random(1)` | `paddle.quantile` |
| `remainder_values` | int `y` 为正值域，其他 dtype default | `paddle.remainder`、`paddle.Tensor.remainder` |
| `dropout_values` | `p` 为 legacy `[0,1.1]` clamp 到 1，其他 Tensor default | `paddle.nn.functional.dropout`、`dropout2d`、`dropout3d` |
| `atan2_values` | `[1,2)` | `paddle.atan2` |
| `bincount_values` | `x/minlength` 为 legacy integer `[0,65535)`，非 int 提前 fallback | `paddle.bincount` |
| `adaptive_avg_pool_values` | `output_size` 上界来自 `x.shape` | `paddle.nn.functional.adaptive_avg_pool2d`、`adaptive_avg_pool3d` |
| `empty_values` | `shape` 为 `[1,10)`，非 int shape Tensor 按 legacy 改写为 `int32` | `paddle.empty` |
| `repeat_interleave_values` | `repeats` 为 `[1,2048)`，`axis` Tensor 提前 fallback | `paddle.repeat_interleave`、`paddle.Tensor.repeat_interleave` |
| `matrix_transpose_values` | `x` 为 legacy default-like 值域，rank<2 时生成 `[2,2]` 数组 | `paddle.matrix_transpose` |
| `softmax_values` | `x` 复刻 `get_random_numpy_tensor()`，`axis` 范围来自 `x.shape` | `paddle.nn.functional.softmax` |
| `zeropad2d_values` | `x` 复刻 `get_random_numpy_tensor()`，`padding` 为 `[0,10)` | `paddle.nn.functional.zeropad2d` |
| `pad_values` | `pad` 范围为 `[0,min(x.shape))`，其他 Tensor default | `paddle.nn.functional.pad` |
| `class_center_sample_values` | `label` 范围为 `[0,num_classes)` | `paddle.nn.functional.class_center_sample` |
| `shard_index_values` | `input` 范围为 `[0,index_num)`，缺省 `index_num` 保留 legacy 随机上界 | `paddle.shard_index` |
| `masked_multihead_attention_values` | `sequence_lengths` 复刻 legacy helper `[1,65535)`，`rotary_tensor` 为 `[0,1000)` | `paddle.incubate.nn.functional.masked_multihead_attention` |
| `argminmax_values` | `axis` 范围来自 `x.shape` 并按 legacy 改写为 `int64` | `paddle.argmax`、`paddle.argmin`、Tensor method |
| `cumsum_values` | `axis` 为 `[-rank,rank)` no-cast | `paddle.cumsum`、`paddle.Tensor.cumsum` |
| `reduction_axis_values` | `axis` 复刻 legacy `generate_random_axes()` | `paddle.mean/max/min/prod/sum/squeeze` |
| `unsqueeze_values` | `axis` 范围来自 `len(x.shape)+1` | `paddle.unsqueeze` |
| `unflatten_values` | `axis` 范围来自 `x.shape`，`shape` Tensor 仍早回退 | `paddle.unflatten`、`paddle.Tensor.unflatten` |
| `topk_values` | `x` 按 dtype 专属值域，`k` 上界来自 `x.shape[axis]` | `paddle.topk`、`paddle.Tensor.topk` |
| `index/gather/take family` | index 范围来自输入 shape/axis，保留 dtype 改写和边界值注入 | `index_sample/index_add/index_fill/take/gather/gather_nd/index_select/take_along_axis` 及 Tensor method |
| `embedding_values` | ids 先消耗 legacy vocab 随机上界，再按 weight shape 取范围；weight 为 `[0,1)`/complex 双 RNG | `paddle.nn.functional.embedding` |
| `loss label family` | label/path/length 范围来自类别数、cutoffs 或输入 shape | `adaptive_log_softmax_with_loss/cross_entropy/ctc_loss/hsigmoid_loss/margin_cross_entropy/multi_margin_loss/dice_loss/nll_loss/softmax_with_cross_entropy/sequence_mask` |
| `clip_values` | 先生成 `min/max` 关系，再生成 `x`，保留 legacy RNG 顺序 | `paddle.clip`、`paddle.Tensor.clip` |
| `cholesky_values` | 生成 SPD 矩阵并保留 legacy stdout | `paddle.linalg.cholesky` |
| `linalg_default_values` | legacy `paddle.linalg.` 大分支中无独立约束的 API 继续使用 default 值域 | `matrix_norm/matrix_rank/lu/multi_dot/norm/matrix_transpose/matrix_power/svd/svdvals/eig/eigvals/svd_lowrank/solve/triangular_solve/inv/qr/vector_norm` |
| `pinv_values` | 仅第 3 个位置参数 `hermitian=True` 时生成 Hermitian 输入；kwarg `hermitian=True` 复刻 legacy typo 保持 default | `paddle.linalg.pinv` |
| `view_values` | uint8 reinterpret 为有限 float/bfloat16 bit pattern，其他 Tensor default | `paddle.view`、`paddle.Tensor.view` |
| `pow_values` | 根据常量底数/指数和 dtype max 收缩随机范围 | `paddle.pow`、`paddle.Tensor.pow` |
| `rnnt_loss_values` | 复刻 logits/label/length 的 legacy shape 修正和固定长度 | `paddle.nn.functional.rnnt_loss` |
| `multinomial_values` | 先生成概率输入，再按正值个数生成 `num_samples` | `paddle.multinomial` |
| `one_hot_values` | 先消耗 legacy 默认类别随机数，必要时写入 `num_classes` Tensor，再生成 `x` | `paddle.nn.functional.one_hot` |
| `chunk_values` | 从可被 `chunks` 整除的维度中用 Python `random.choice` 选择 axis | `paddle.chunk` |
| `split_values` | 根据 `num_or_sections` 选择合法 axis | `paddle.split` |
| `expand_values` | `shape` Tensor 根据 `x.shape` 与 shape 参数位置生成 | `paddle.expand`、`paddle.Tensor.expand` |
| `gather_tree_values` | `parents` 范围来自 beam size | `paddle.nn.functional.gather_tree` |

这些规则仅在 v2、CPU、未开启 NumPy cache 时执行。默认 legacy 行为没有改变。

部分规则使用 `case.generate("参数名", "generator")` 选择参数值域；未命中特定参数的
Tensor 可用 `case.generate_remaining("default")` 按 legacy 尾部分支补齐，并保持原遍历
顺序。大多数规则不读取其他 Tensor 的已生成值；少数 legacy 语义要求 dtype/shape mutation
的规则会在写回 NumPy 数组时同步 `TensorConfig.dtype/shape`，例如 `empty`、`matrix_transpose`
和 `adaptive_log_softmax_with_loss` scalar label。

当前没有建立 positive/unique 通用 rule：

- 其他 positive 分支通常读取其他参数或使用 API 专属范围，不能合并为一个无状态 generator；
  本批次只迁移了范围和 helper 调用完全固定的 sqrt/rsqrt。
- `generate_unique_array` 当前没有可直接迁移的独立运行时调用点。

保持 legacy 比建立语义不准确的抽象更安全。后续只有在找到精确、单一且可对照的
调用族后才注册新规则。

## 9. 文件职责与生命周期

### 9.1 运行时模块

| 文件 | 当前职责 | 长期状态 |
|---|---|---|
| `parser.py` | `APIConfig` 和配置解析 | 保留，后续单独治理 parser |
| `tensor_config.py` | Tensor 配置、cache、框架物化 | 保留但继续拆分 |
| `input_generator.py` | 原 legacy API 条件链 | 临时，fallback 归零后删除 |
| `model.py` | 不可变路径、绑定和上下文模型 | 保留 |
| `payload.py` | `TensorPayload` 逻辑值容器和 `api_config` payload 存取 | 保留；作为 rule 输出真源，继续收窄 `TensorConfig.numpy_tensor` 回退面 |
| `binding.py` | API 签名与参数路径绑定 | 保留并逐步成为唯一绑定实现 |
| `strategies.py` | 旧工具兼容导出 shim | 仅兼容保留；新代码不应从此导入 |
| `registry.py` | decorator rule、RuleCase、API 查找和冲突检测 | 保留 |
| `dispatcher.py` | legacy/v2 选择和完整 case fallback | 迁移期保留；legacy 删除后简化 |
| `telemetry.py` | context-local dispatch/legacy 事件 | 保留，接入 CI 报告后可扩展 |

当前文件数量来自职责边界，而不是按 API 拆文件。现阶段不会为每个 value generator
创建一个模块。随着 decorator rule 增长，后续应按 API 领域拆成
`rules/elementwise.py`、`rules/indexing.py`、`rules/loss.py` 等模块；这些模块仍通过
同一个 `rules.register()` 注册，不再引入第二种静态表格式。

### 9.2 离线治理工具

`tools/input_generation_governance/` 不被生产运行时导入：

| 资产 | 用途 |
|---|---|
| `legacy_rule_inventory.json` | 枚举 legacy 分支、依赖和源码摘要 |
| `baseline_cases.yaml` | 固定 seed 的代表性 case 清单 |
| `legacy_cpu_baseline.json` | CPU dtype/shape/bytes 基线 |
| `verify_strategies.py` | value generator 与 legacy 的 bytes/RNG 对照 |
| `verify_registry.py` | registry、dispatch、fallback 和端到端 v2 对照 |
| `verify_bindings.py` | 新旧参数绑定 shadow 对照 |
| `verify_input_generation.py` | inventory、CPU/GPU 基线入口 |

这些文件是迁移的证据和门禁，不属于输入生成运行时。legacy 删除后，inventory 可以
移除；固定 seed、binding、registry 和策略验证仍应作为长期回归工具保留或转入 CI。

## 10. 语义一致性约束

一次迁移只有同时满足以下条件，才能加入 production registry：

1. 固定 seed 下数组 dtype、shape 和 bytes 一致。
2. 固定 seed 下最终 NumPy RNG state 一致。
3. Tensor 遍历顺序与 legacy 一致。
4. scalar、empty、normal shape 均有覆盖。
5. 相关整数、浮点、复数、BF16/FP8 中间 dtype 均有覆盖。
6. 未迁移 API 的 legacy telemetry 和生成结果不变。
7. GPU/cache 未支持时，在任何新 RNG 调用前回退。
8. default runtime 仍为 legacy，除非单独评审切换计划。

`CaseNumpyRNG` 当前是 case 级 RNG facade，记录 seed、config fingerprint 和 runtime mode。
它的 backend 仍是 `legacy-global`，随机方法继续代理全局 `numpy.random`，因此没有改变
随机算法、调用顺序或最终 RNG state。后续切换真正独立的 case-owned RNG state 时，必须
单独验证固定 seed bytes 与最终 RNG state。

## 11. API 迁移流程

每批迁移按以下步骤执行：

1. 从 inventory 和源码确认 legacy 分支的完整选择条件。
2. 检查是否读取其他参数、写入其他 Tensor、依赖已初始化值、GPU、cache 或 backend。
3. 只有无关系的 dtype/shape/value domain 才先抽为 value generator。
4. 使用 shadow verifier 对照数组 bytes 和最终 RNG state。
5. 为经过确认的 API 建立显式 allowlist，不使用名称前缀猜测。
6. 定义不支持模式的 `fallback_reason`，且 fallback 必须早于任何 RNG 调用。
7. 执行真实 `APITestBase.gen_numpy_input()` 的 legacy/v2 固定 seed 对照。
8. 检查迁移 case 的 legacy events 归零，未迁移 case 的 events 保持一致。
9. 执行 CPU baseline、binding、已有测试、GPU verifier 和 pre-commit。
10. 更新计划、发现和进度文档。

以下情况不得进入基础策略迁移批次：

- 生成一个参数时写入另一个参数。
- 依赖其他 Tensor 已经完成生成。
- 改变 shape、dtype、place 或配置对象附加属性。
- 直接创建 Paddle/Torch/GPU Tensor。
- 依赖 optimizer flag、cache 内容、FP8 打包或 DLPack。
- 无法证明所有随机调用顺序一致。

## 12. 验证与可观测性

当前主要验证命令：

```bash
python -m tools.input_generation_governance.verify_strategies
python -m tools.input_generation_governance.verify_registry
python -m tools.input_generation_governance.verify_input_generation --mode cpu
python -m tools.input_generation_governance.verify_bindings --strict
python -m tools.input_generation_governance.verify_bindings \
  --config-root tester/api_config --limit 500
python -m tools.input_generation_governance.verify_input_generation \
  --mode gpu --require-gpu
```

GPU 不可用时，普通 GPU verifier 必须报告 `SKIP`，不能报告通过；GPU CI 使用
`--require-gpu` 把无设备视为失败。

Telemetry 当前使用 `ContextVar`，只有验证或调用方显式 capture 时收集事件。正常运行
不会维护进程级累计列表，也不会消耗 RNG。事件至少能够区分：

- legacy 默认执行。
- v2 rule 命中。
- v2 未注册回退。
- v2 因 GPU/cache 等原因回退。

## 13. 后续演进顺序

1. 完成基础 value generator 审计，只迁移可证明同构的单参数规则。
2. 按 creation/random、elementwise、reduction 等低风险 family 小批迁移。
3. 增强 `RuleCase` 的 payload、引用和 validator 能力，迁移 indexing、shape、loss 等规则。
4. 按领域迁移 linalg、vision、optimizer、attention、MoE 和 FP8。
5. 将 NumPy payload 与 Paddle/Torch/GPU materializer 完全拆分。
6. 将全局 RNG 改为 case-owned RNG，将全局 cache 改为 session-owned cache。
7. 单独治理 parser/serializer。
8. 当核心语料和 CI 中 fallback 持续为零后，删除 `input_generator.py` 和双轨开关。

每个阶段只处理一类风险。规则迁移、随机分布调整、RNG 算法切换和 materializer
改造不能合并在同一个变更中。

## 14. 常见疑问

### 为什么不直接重写 legacy 条件链？

因为无法一次性证明数千行分支、跨参数副作用和全局 RNG 顺序完全一致。双轨允许每批
只承担一类风险，并让任何未覆盖 API 明确回退。

### 为什么保留 `input_generator.py`？

它是当前未迁移 API 的行为真源和回退实现。过早删除会迫使高风险规则与基础设施一起
迁移。只有 fallback 稳定归零后才删除。

### 为什么 default value generator 只注册两个 API？

value generator 本身适用于普通 dtype/shape，但不能证明所有未列出的 API 都会到达 legacy
默认分支。allowlist 表示“已证明安全”，而不是“理论上可能适用”。

### 为什么 rule 以完整 case 为单位？

因为很多约束属于参数之间的关系。逐 Tensor rule 无法消除生成顺序依赖，只会把旧的
副作用模型换一种写法保存下来。即使当前 rule 只调用 `case.generate_all("default")`，
它也已经在完整 case 边界执行；后续复杂 rule 只是在同一个形态下增加参数引用和
validator。

### 为什么当前规则仍回写 `TensorConfig`？

现有规则仍同步回写 `TensorConfig.numpy_tensor`，这是保持外部协议不变的回退桥。
`RuleCase` 现在已经记录独立 `TensorPayload`，`RegisteredRule` 会把 payload 挂到
`api_config`，Paddle/Torch materializer 优先消费 payload；decorator rule 的 API 不应
随这个内部所有权变化而大改。

### 为什么治理工具不放在运行时包？

inventory、基线 JSON 和 verifier 只用于审查与 CI，不应增加生产 import 成本，也不应
成为运行时 API。运行时和迁移证据使用独立目录，但通过固定 schema 互相校验。

## 15. 完成标准

输入生成治理最终完成时应满足：

- production registry 覆盖全部目标 API，legacy fallback 持续为零。
- 核心生成层不再新增 `api_name` 条件链。
- 所有跨参数关系由完整 case rule 和 validator 显式表达。
- Paddle/Torch/GPU 从同一逻辑 payload 物化。
- 相同 case fingerprint 和 seed 可独立复现，不受执行顺序影响。
- cache 不共享可写数组，容量和生命周期明确。
- 删除 legacy generator、迁移期开关和回写兼容层。
- CI 能报告 rule 命中、fallback、seed、ArgPath、dtype/shape 和 materializer。

在达到这些条件前，当前双轨设计是风险控制机制，不是最终架构本身。
