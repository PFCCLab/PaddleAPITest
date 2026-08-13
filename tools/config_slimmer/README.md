# Config Slimmer

`config_slimmer` 对一个或多个逐行 API 配置集进行联合瘦身。它先做轻量 API 分布统计，再按文件和结构族分批提取特征、选择代表配置；跨文件完全重复的配置仍只保留一次。工具不执行配置文本，默认提高复杂 API、自定义 API 的保留率，并对简单 API 进行更积极的缩减。

目录中还提供独立的 `conservative_slim_configs`。它通过显式 API 模型注册表决定每个 API 的选择策略，不根据名称正则或统一复杂度评分自动套用策略；原有 `slim_configs` 的行为和参数不受影响。

## 保守近重复精简

注册表位于 `api_models.py`。目标 `1M_preprocessed.txt` 中的 118 个 API 均有明确模型，未注册的新 API 默认全量保留。当前模型如下：

| 模型 | 最低保留率 | 适用范围 |
|---|---:|---|
| preserve | 100% | transpose、索引、数据重排、随机、归约等复杂内核 |
| numeric_near | 至少 80% | 只允许完整数值向量近邻合并的 API |
| simple_coverage | 10% | 创建、填充和简单元数据类内核 |
| cast_coverage | 20% | cast/astype，dtype 组合严格隔离后覆盖长度与对齐 |
| elementwise_coverage | 40% | 不涉及广播的逐元素内核 |
| broadcast_coverage | 70% | 二元、原地和广播逐元素内核 |
| linear_algebra_coverage | 75% | matmul、addmm、baddbmm、linear、einsum 等 |
| custom_coverage | 70% | 已建模的大规模 custom op |

各 API 可以共享选择器，但 API 到模型的归属是逐项登记的。Custom op 还会按字符串名称再次解析模型；未登记的 custom op 默认保留。

`numeric_near` 模型对所有非字符串数值位置执行通用近重复判断：

```bash
python -m tools.config_slimmer.conservative_slim_configs \
  /path/configs.txt \
  --output-dir /path/conservative_slimmed
```

保守策略先按严格的非数值文本和数值类型分组。API 名、参数形式、字符串、dtype、布尔值或任意其他非数值内容不同的配置不会进入同一组。组内两条配置只有在每个数值位置均满足以下条件时才算近重复：

```text
absolute_delta <= max_absolute_delta
absolute_delta <= max(absolute_tolerance, relative_tolerance * max(abs(a), abs(b)))
```

默认相对阈值为 2%，最大绝对差值为 256；绝对值不超过 16 的整数保持精确，绝对值达到 `1e9` 的整数也保持精确，并且不跨符号或 `log2` 量级边界。每个结构组最多删除 20%，小于 20 条的组不处理。每个删除项必须直接匹配一个保留代表，不使用可放大范围的链式相似关系。

默认还会保留每个数值位置的最小值、最大值、量级、对齐类别和 2 的幂邻域代表。主要调整参数如下：

- `--relative-tolerance`：逐位置相对差值，默认 `0.02`。
- `--absolute-tolerance`：接近零时使用的绝对容差，默认 `0`。
- `--max-absolute-delta`：任何位置允许的最大绝对差值，默认 `256`。
- `--exact-small-integer`：不参与近似的小整数范围，默认 `16`。
- `--exact-integer-above`：从该值开始按精确大整数处理，默认 `1000000000`。
- `--max-removal-rate`：每个结构组的删除率硬上限，默认 `0.20`。
- `--min-group-size`：允许处理的最小结构组，默认 `20`。
- `--preserve-api` 和 `--pin-file`：强制保留指定 API 或完整配置行。
- `--dry-run`：只输出预计统计，不写文件。

复杂 API coverage 模型使用显式 profile，同时仍受严格结构分组和模型最低保留率约束。当前 profile 包括：

- `moe_permute`：保持 Tensor shape 的联动关系；`tokens_per_expert` 长度和零值数量不变，排序后逐项默认允许 20% 且不超过 1024 的差值，总 token 数差不超过 5%。排序比较避免仅因 expert 槽位交换而保留重复负载分布。
- `moe_unpermute`：原始 token 数和 zipped token 数使用通用近邻阈值，并要求 zipped token 数在 Tensor shape 和标量参数中的等值关系保持不变。
- `fp8_quant_blockwise`：覆盖 row/column block 数量级、128 对齐余数、dtype、quant method 和 transpose/scale 开关。
- `fused_act_dequant`：覆盖各 shape 位置的数量级、dtype、rank、Tensor 关系和数值边界。
- `_run_custom_op`：先按 custom op 名和调用 schema 隔离，再覆盖 Tensor shape/dtype/rank、参数关系和数值数量级。

相关参数：

- `--no-complex-api-profiles`：关闭全部复杂 API profile，恢复完整数值向量逐位置比较。
- `--moe-sequence-relative-tolerance`：MoE 负载序列逐项相对阈值，默认 `0.20`。
- `--moe-sequence-max-absolute-delta`：MoE 负载序列逐项最大绝对差，默认 `1024`。
- `--moe-sequence-sum-relative-tolerance`：MoE 负载序列总量相对阈值，默认 `0.05`。

### 简单内核精简

注册为 `simple_coverage` 的 API 使用 coverage-aware 抽样：detach/clone、item/tolist/numel/dim、zeros/full/empty、assign/zero_ 和 arange。执行代码中固定最低保留率为 10%，每个结构族至少保留 4 条；为了覆盖 dtype、rank、数值区间、对齐、幂边界和极值，实际保留率可以更高。cast/astype 单独保留至少 20%，且不同源/目标 dtype、参数形式不会混合。

`transpose/transpose_` 虽然 API 形式简单，但内核路径复杂，内置全量保护，包括 `paddle.transpose`、Tensor 方法和 `_C_ops` 形式。相同处理还覆盖 getitem/setitem/gather/embedding/where、concat/cat/stack/split/chunk/unbind/pad、broadcast/expand/nonzero/repeat_interleave、随机生成、dropout、normalize/softmax/rms_norm、reduction、adamw 和 SwiGLU 前反向融合。reshape/view/flatten/squeeze/unsqueeze 主要是 shape/view 语义，不执行 transpose 式通用数据置换，使用严格数值近邻模型，每个结构组最多删除 20%。矩阵类与广播逐元素类分别至少保留 75% 和 70%。

报告提供 `by_api`、`by_model`、`unmodeled_apis`、`excluded_near_duplicates` 和 `excluded_coverage_cases`。审计 TSV 为每条配置记录实际采用的模型。

保守策略单独生成：

- `<原文件名>_conservative_deduplicated.txt`
- `<原文件名>_conservative_slim.txt`
- `<原文件名>_conservative_excluded.txt`
- `conservative_report.json`
- `conservative_decisions.tsv`

审计 TSV 为每条被删除配置记录保留代表、发生变化的数值位置、最大绝对差值、最大相对差值和序列总量相对差值。复杂 API 的审计距离使用与 profile 相同的排序或联动比较方式。

## 快速使用

在 `PaddleAPITest` 根目录执行：

```bash
python -m tools.config_slimmer.slim_configs \
  /path/api_config_1.txt /path/api_config_2.txt \
  --output-dir /path/slimmed \
  --progress
```

默认生成：

- `<原文件名>_deduplicated.txt`：预处理后的有序唯一配置。
- `<原文件名>_slim.txt`：最终保留的有序配置。
- `<原文件名>_excluded.txt`：去重后被 slimmer 排除的唯一配置。
- `coverage_report.json`：整体、各文件、各 API 的保留统计及建模特征覆盖率。
- `decisions.tsv`：每条唯一配置的决定、重复次数、优先级，以及被排除项对应的保留代表项 ID。

已有输出不会被覆盖。确认需要覆盖时显式传入 `--force`。

三个配置输出默认按整行字典序排序，相同输入和参数始终产生相同顺序。需要兼容原始首次出现顺序时，可显式传入 `--preserve-input-order`。

## 默认策略

| 优先级 | 自动识别范围 | 保留率 | 单结构族最少保留 |
|---|---|---:|---:|
| custom | `_run_custom_op` 及用户指定的自定义 API | 50% | 64 |
| high | 融合、MoE、量化、多 Tensor/多参数复杂 API | 35% | 32 |
| medium | 矩阵、归一化、聚合、组合等 API | 20% | 12 |
| low | 简单形变、创建、查询、逐元素 API | 8% | 4 |

结构族由 API、调用形式、字符串参数、dtype 等不应混淆的属性组成。族内选择覆盖 Tensor rank、shape 关系、广播关系、对齐、幂边界、数值分位、最小值、最大值、布尔开关和数值序列偏斜等特征，然后按固定 seed 补足预算。

无法识别 API 名的非空行会完整保留，不会静默删除。多个输入中的完全重复配置会在全局保留一次；注释和空行不参与统计和抽样，并随配置一起排序输出。

对于每个输入文件，配置部分满足：

```text
deduplicated = slim + excluded
```

原文件中的重复出现次数记录在 JSON 报告和 TSV 审计中，不会重复写入 `_excluded.txt`。

## 调整优先级

参数接受 Python 正则表达式，可重复指定：

```bash
python -m tools.config_slimmer.slim_configs configs.txt \
  --high-api 'paddle\.my_complex_api$' \
  --custom-api 'paddle\.my_custom_api$' \
  --low-api 'paddle\.(?:zeros|full)$' \
  --preserve-api 'paddle\.experimental\.' \
  --custom-rate 0.6 --high-rate 0.4 --medium-rate 0.2 --low-rate 0.05
```

- `--custom-api/--high-api/--medium-api/--low-api`：覆盖自动优先级。
- `--preserve-api`：匹配 API 的所有配置均保留。
- `--pin-file`：文件中的完整配置行必须保留，可用于历史失败和回归用例。
- `--min-custom/--min-high/--min-medium/--min-low`：调整每个结构族的最少保留数。
- `--slim-strength`：整体精简力度，范围 0 到 1；`0` 只按最小保留数兜底，`1` 使用默认保留率。
- `--progress`：向 stderr 打印扫描、读入、大结构族选择进度，适合十万级以上配置集。
- `--seed`：控制确定性选样；输入和参数不变时结果不变。

## 只查看预计结果

```bash
python -m tools.config_slimmer.slim_configs configs.txt --dry-run
```

`--dry-run` 向标准输出打印 JSON 统计，不创建文件。
