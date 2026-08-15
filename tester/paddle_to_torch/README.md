# Paddle2Torch

## 目录

- [模块介绍](#模块介绍)
- [开发文档](#开发文档)
- [贡献指南](#贡献指南)
- [高级Rule指南](#高级Rule指南)
- [结语](#结语)

## 模块介绍

Paddle2Torch 是一个专注于将 PaddlePaddle API 转换为 PyTorch 对应实现的知识工具库，属于 [PaddleAPITest](https://github.com/PFCCLab/PaddleAPITest) 项目的核心组成模块。本模块通过解析 PaddlePaddle API 调用，使用预定义的转换规则与动态代码生成，实现从 PaddlePaddle 到 PyTorch 的自动转换。转换过程将确保代码的语义一致性。

本模块具有精简强悍的架构，仅由三个组件构成：
- *转换引擎 converter.py*
- *转换配置 mapping.json*
- *转换规则 rules.py*

代码已完全进行解耦，可以非常容易地迁移至其他代码中。本模块通过 **转换配置** 与 **转换规则** 管理 API 映射关系，因此支持开发者灵活扩展新的 API 转换能力。

本模块的典型应用场景包括：模型迁移、跨框架验证、混合编程等，可为深度学习开发者提供跨框架的互操作性解决方案。

## 运行时配置与转换状态

Paddle2Torch 统一通过 `PADDLEAPITEST_IMPL` 选择参考实现。合法值为 `torch`、`te` 和
`apex`。未设置时，每个 Rule 使用自身声明的默认实现；融合线性梯度 Rule 默认为 `te`，
其余多实现 Rule 默认为 `torch`。当全局值对当前 Rule 不适用时，该 Rule 同样使用自己的
默认实现；非法全局值会在转换前明确报错。
转换引擎对每次调用只读取一次配置快照，同一快照同时用于生成代码和转换缓存键。
原融合线性专用实现变量已退场，不再读取；相关作业必须改用 `PADDLEAPITEST_IMPL`。

`PADDLEAPITEST_WORKERS_ON_GPU` 用于划分单个 GPU 上每个 worker 可使用的临时 workspace，
默认值为 `1`，并且必须是正整数。该配置在 workspace 计算时校验，不影响转换代码缓存。

`mapping.json` 在转换器初始化时进行 schema 校验，并拒绝重复 JSON key。所有 API 名必须以 `paddle.` 开头；
允许的字段为 `Rule`、`torch_api`、`set_defaults`、`paddle_torch_args_map`、`torch_args`、
`torch_kwargs`、`is_attribute` 和 `description`。转换器按 Rule 的 `PADDLE_APIS` 注册表选择
自定义 Rule；`mapping.json` 中对应条目的 `Rule` 字段是便于按 API 查找类名的索引，必须
与注册表中的实际类名完全一致。没有注册自定义 Rule 的配置不得声明 `Rule`，使用
`GenericRule` 且必须提供非空 `torch_api`。默认完整 mapping 还必须覆盖注册表中的每个
API。参数映射的键值也必须符合 schema。未知字段、错误字段类型及 Rule 索引不一致会在
初始化阶段报告具体 API 与字段；校验完成后的 mapping 和 Rule registry 均为只读。

`ConvertResult.kind` 使用 `ConversionKind.UNSUPPORTED`、`ConversionKind.DIRECT` 和
`ConversionKind.COMPOSITE` 表示不支持、直接 Torch 对应和组合实现。仅 `DIRECT` 适用于
直接性能对比，`COMPOSITE` 仍是可执行的受支持转换。`Code` 和 `ConvertResult` 均不可变。
转换阶段异常会包含 Paddle API 和 Rule 类，执行阶段异常会包含 Paddle API 及
`preprocess`、`core` 或 `postprocess` 阶段。

## Rule 编写规范

新增 Rule 必须遵守以下规范，存量 Rule 在后续修改时逐步迁移。Rule 按实现方式分为三类：

- **Mapping Rule**：只涉及默认值、参数改名、位置参数或关键字参数调整，必须仅使用
  `mapping.json`，不新增 Rule 类。
- **Adaptation Rule**：核心是一次明确的 Torch API 调用，但调用前后需要 dtype、shape、
  layout 或输出结构适配，使用 `ConversionKind.DIRECT`。
- **Reference Rule**：使用多个 Torch 操作、控制流、自定义数学实现或外部 kernel，使用
  `ConversionKind.COMPOSITE`。

Paddle 签名默认值由共享参数绑定器统一应用。`mapping.json` 的 `set_defaults` 只用于无签名
API 或转换阶段新增的局部变量；Rule 不应再次手写相同默认值或参数映射。
`build_result()` 自动按“默认值、Rule preprocess、参数映射”的固定顺序组装 preprocess。
Rule 只编写自定义参数归一化，且归一化命名参数而不是提前构造或修改 `_kwargs`。

Rule 使用 `build_result()` 构造结果，并显式声明 `DIRECT` 或 `COMPOSITE`。三个代码阶段的
职责固定如下：

- `preprocess`：Rule 自定义的参数归一化、必要 import 和局部辅助函数；默认值和参数映射由
  `build_result()` 自动插入。
- `core`：执行参考实现，并将结果写入 `result`。
- `postprocess`：恢复 Paddle 的输出 dtype、layout 或数据结构。

存在多个参考实现的 Rule 在类上声明 `SUPPORTED_IMPLEMENTATIONS` 和
`DEFAULT_IMPLEMENTATION`，将实现代码生成函数统一命名为 `_<实现名>_code()`，并通过
`build_implementation_code()` 选择实现。Rule 不得直接读取实现环境变量，也不得在显式
选择的外部实现不可用时静默回退。

## 开发文档

百度内部同学请参考：
- [Paddle2Torch 内核机制开发文档](https://ku.baidu-int.com/d/ODBEcpC8QXcAAE)
- [PaddleAPITest Paddle2Torch 使用文档](https://ku.baidu-int.com/d/-75canpiFaJClt)

## 贡献指南

如果您在使用或测试过程中发现尚未支持的 Paddle API 转换，可以参考以下开发流程进行快速开发，完善 Paddle2Torch 的转换能力。以 paddle.crop 为例：

### 检查支持情况

1. 首先在 mapping.json 中搜索 paddle.crop，查看是否已有相关 API 配置。若存在，可以在全局搜索 API 名称，提取其所有测试配置，进行测试；若无任何搜索结果，说明此 Paddle2Torch 尚未支持转换此 API，需要我们补齐转换能力。此时未搜索到 paddle.crop，开始进行补齐工作。 

### 查询开发资料

2. 在 [paddle 官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html) 中搜索 paddle.crop，对照 API 文档，做好转换能力开发的准备。paddle.crop 的 API 介绍为：

    > paddle.crop(x, shape=None, offsets=None, name=None)
    > 
    > 根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor。

    飞桨官方开发了 Torch 转 Paddle 的强大代码工具 [PaConvert](https://github.com/PaddlePaddle/PaConvert) ，并且飞桨文档中也有完备的 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) ，详细说明了哪些 API 可以互相转换，不能转换的原因与可能的解决办法是什么。我们可以先查询并参考这些资料👆。

3. 在 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) 中搜索 paddle.crop，查看是否有符合条目。若存在，则分别点击 **Torch API** 和 **详细对比**，仔细阅读内容，思考其提供的方案是否可行；若没有发现任何条目，说明此 API 是比较少见的类型、或是新 API，需要我们再次仔细阅读 API 文档描述，思考并查询对应的 Torch API 可能是什么。paddle.crop 没有现成的转换方案，需要进一步寻找。


4. 在 [PyTorch 官网](https://pytorch.org/docs/stable/index.html) 中搜索 crop，仅找到图像操作的 API： [torchvision.transforms.functional.crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html) ，不太符合我们想要的 Torch API。

   经查阅资料，能够实现 paddle.crop 表现的有 torch.narrow 或直接使用切片操作（Torch 重载了 [] 操作符）。前者仅能实现单维度裁剪，实现多维度需要进行循环，较为复杂；后者则类似于 numpy 风格的切片，虽然也需要循环，但可以压缩至一行，非常 pythonic。因此决定使用 Torch 的切片操作模拟 paddle.crop 的表现。

### 组织编写思路

5. 由于构造切片所用的 slices 参数需要使用循环，且属于特殊操作（不属于调用 Torch API），因此需要继承 BaseRule，编写新的 Rule 类。如果能够通过 **直接参数映射** 或 **组合映射** 方式实现的话，建议最好在 mapping.json 中编写配置即可，可直接跳转至 [编写转换配置](#编写转换配置) 章节。

6. paddle.crop 的参数介绍中详细介绍了不同参数的类型、默认值等，我们需要支持所有的配置情况，并考虑到参数缺省。参数介绍如下：

    > **x** (Tensor) - 1-D 到 6-D Tensor，数据类型为 float32、float64、int32 或者 int64。
    > 
    > **shape** (list|tuple|Tensor，可选) - 输出 Tensor 的形状，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应该是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代时需要改变输出形状的情况。
    > 
    > **offsets** (list|tuple|Tensor，可选) - 每个维度上裁剪的偏移量，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代的偏移量（offset）都可能改变的情况。默认值：None，每个维度的偏移量为 0。

    可以看到，paddle.crop 的 shape、offsets 参数具有非常丰富的形式，可以是 *缺省*、*列表或元组*、*1-D Tensor*，列表或元组可以由 *int* 或 *0-D Tensor* 组成。

7. 在测试配置中搜索 paddle.crop ，可以看到 shape 中允许 -1，说明该维度的大小由 x 和 offsets 推断，我们也需要支持此种配置。

    ```python
    paddle.crop(
        x=Tensor([2, 3, 3, 3], "float64"),
        shape=list[
            2,
            1,
            -1,
            2,
        ],
        offsets=list[
            0,
            0,
            1,
            1,
        ],
    )
    ```

### 编写转换代码

8. 在编写代码前，共享参数绑定器已经根据 Paddle 签名将位置参数和关键字参数统一绑定，
   并以 Paddle 参数名放入执行环境 `locals()`。新 Rule 直接读取必需参数；可选参数应在
   mapping 的 `set_defaults` 中声明后直接读取。未提供的必需参数直接访问时会抛出
   `NameError`。原始 `args`、`kwargs` 仅作为尚未迁移完成的内部执行桥，不属于 Rule
   编写接口。

    首先单独构造出 slices 可用的 shape 与 offsets 参数，使用 list 来表示（默认所有参数均是符合文档描述的，不需要再验证和抛出错误）：

    ```python
    ndim = x.dim()

    if offsets is None:
        offsets = [0] * ndim
    elif isinstance(offsets, (list, tuple)):
        offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
    elif isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()

    if shape is None:
        shape = [x.size(i) - offsets[i] for i in range(ndim)]
    elif isinstance(shape, (list, tuple)):
        shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
    elif isinstance(shape, torch.Tensor):
        shape = shape.tolist()
    ```

    推断并替换 shape 中所有 -1 值。

    ```python
    shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
    ```

    根据 shape 与 offsets 构造 slices 参数：

    ```python
    slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]
    ```

    使用 Torch 切片操作，将结果保存至 result 中（ x 一定存在于 `locals()` 中，不需要再使用 `get()` ）：

    ```python
    result = x[tuple(slices)]
    ```

    至此，转换代码编写完成.

### 测试转换代码

9. 为了验证转换代码的正确性，我们可以编写一些简单的测试用例去测试它，避免到了测试执行时才报错：

    ```python
    import torch


    def torch_crop(x, shape=None, offsets=None):
        ndim = x.dim()
        if offsets is None:
            offsets = [0] * ndim
        elif isinstance(offsets, (list, tuple)):
            offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
        elif isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        if shape is None:
            shape = [x.size(i) - offsets[i] for i in range(ndim)]
        elif isinstance(shape, (list, tuple)):
            shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
        elif isinstance(shape, torch.Tensor):
            shape = shape.tolist()

        shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
        slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]

        return x[tuple(slices)]


    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, [2, 2], [1, 1]))

    x = torch.arange(27).reshape(3, 3, 3)
    print(torch_crop(x, [-1, 2, 2], [0, 1, 0]))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, torch.tensor([2, 2]), torch.tensor([1, 1])))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x, [torch.tensor(2), 2], [torch.tensor(1), 1]))

    x = torch.arange(16).reshape(4, 4)
    print(torch_crop(x))
    ```

    测试结果符合预期，我们成功地使用了 Torch 模拟出 Paddle API 的所有表现了！现在可以开始编写 Rule 类了！

### 编写转换配置

10. 若仅需要编写转换配置，需在 mapping.json 的相应条目（去掉 paddle. 后的字典序）下编写，编写规则为：

    ```json
        "<api_name>": {
            "torch_api": "torch api 名称",
            "set_defaults":{
                "_description1": "默认值设置字典，键为参数名，值为默认值",
                "_description2": "仅在命名参数缺失时赋值，不覆盖调用方传入值"
            },
            "paddle_torch_args_map": {
                "_description": "参数名映射字典，键对应 paddle，值对应 torch",
            },
            "torch_args": [
                "torch api 位置参数列表, 变量名可使用 {} 环绕，字符串的引号请使用 \\ 转义，可以直接设为常值"
            ],
            "torch_kwargs": {
                "_description": "torch api 关键字参数字典，与 torch_args 类似"
            }
        }
    ```

11. 若需要编写转换代码，在 `rules.py` 中定义类并通过 `PADDLE_APIS` 声明精确 API；
    同时在 `mapping.json` 的 API 条目中填写同名 `Rule` 字段，作为从 API 定位实现类的
    检索索引。转换器运行时仍直接按 API 注册表选择 Rule，并在初始化时校验两者一致。

    此外，也可以添加更多的常规配置，以减少 Rule 类代码的编写量。`build_result()` 会
    自动把默认值与参数映射加入 preprocess：

    ```json
        "<api_name>": {
            "Rule": "CropRule",
            "torch_api": "torch api 名称",
            "set_defaults":{
                "_description1": "默认值设置字典，键为参数名，值为默认值",
                "_description2": "仅在命名参数缺失时赋值，不覆盖调用方传入值"
            },
            "paddle_torch_args_map": {
                "_description": "参数名映射字典，键对应 paddle，值对应 torch"
            }
        }
    ```

    对于 `paddle.crop`，Rule 直接声明 API 所有权：

    ```python
    class CropRule(BaseRule):
        PADDLE_APIS = ("paddle.crop",)
    ```

    因此 `paddle.crop` 条目的 `Rule` 必须为 `"CropRule"`。重命名 Rule 或调整
    `PADDLE_APIS` 时必须同步更新 mapping；不一致会在转换器初始化时直接失败。
12. Rule 类需要继承 BaseRule、声明 `PADDLE_APIS` 并实现 `apply()` 方法，否则无法执行转换。基类定义为：

    ```python
    class BaseRule(ABC):
    """转换规则的抽象基类"""

    PADDLE_APIS: tuple[str, ...] = ()

    @abstractmethod
    def apply(self, paddle_api: str) -> ConvertResult:
        pass
    ```

    在 rules.py 的 #c 注释下编写 Rule 类 CropRule：

    ```python
    class CropRule(BaseRule):
        PADDLE_APIS = ("paddle.crop",)

        def apply(self, paddle_api: str) -> ConvertResult:
            core = """
    ndim = x.dim()
    if offsets is None:
        offsets = [0] * ndim
    elif isinstance(offsets, (list, tuple)):
        offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
    elif isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    if shape is None:
        shape = [x.size(i) - offsets[i] for i in range(ndim)]
    elif isinstance(shape, (list, tuple)):
        shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
    elif isinstance(shape, torch.Tensor):
        shape = shape.tolist()
    shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
    slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]
    result = x[tuple(slices)]
    """
            return self.build_result(
                paddle_api,
                kind=ConversionKind.COMPOSITE,
                core=core,
            )
    ```

### 运行测试配置

13. 全局搜索 paddle.crop ，将所有相关测试配置移至临时文件中，然后运行 accuracy 测试命令：

    ```shell
    python engine.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt"
    ```

    最终测试配置全部通过，结果位于 test_log/api_config_pass.txt，合并至通过 accuracy 测试的 api_config_support2torch_*.txt 中。

### 其他情况

14. 如果 Paddle API 的行为实在难以通过 Torch 表达，可暂时不注册 mapping，并将所有相关配置合并至未通过 accuracy 测试的 api_config_paddleonly_*.txt 中。

## 高级Rule指南

在最新版本的 Paddle2Torch 中，我们引入了更高级的 Rule 编写方式，可以更方便地处理复杂情况。包括减少编码量，提高可读性，并且有利于实施后续的 Paddle API 性能测试。以 paddle.nn.functional.conv1d 为例：
```
paddle.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format='NCL', name=None)
```

1. 查询对照表：
   
在 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html) 中搜索 	paddle.nn.functional.conv1d，找到 [torch.nn.functional.conv1d 与 paddle.nn.functional.conv1d 对照表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/functional/torch.nn.functional.conv1d.html)，发现是 paddle 参数更多，需要注册并编写 Conv1dRule 类

2. 查阅文档：

查阅 [Paddle 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/conv1d_cn.html) 与 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html?highlight=conv1d#torch.nn.functional.conv1d) 后发现参数差异：

**paddle 参数更多**：paddle 多支持 data_format 参数，需使用 permute 调换输入与输出维度顺序

**paddle 与 torch 的参数用法不同**：stride、dilation 若为列表，需转换为 tuple 类型

**padding 参数形式更丰富**：
- 当 padding 为 “SAME” 或 “VALID” 时，torch 也支持此设置，直接转写为小写
- 当 padding 为长度为 1 的列表时，转为 tuple 类型
- 当 padding 为长度为 2 的列表时，代表非对称填充，torch 对应的 api 不支持非对称填充，因此需使用 torch.nn.functional.pad 对 torch 的输入进行手动填充

3. 编写转换配置
   
因为 paddle.nn.functional.conv1d 参数较多，因此在 mapping.json 中编写转换层默认值与参数映射表，减少 Rule 类编写量：

```json
    "paddle.nn.functional.conv1d": {
        "Rule": "Conv1dRule",
        "torch_api": "torch.nn.functional.conv1d",
        "set_defaults": {
            "bias": "None",
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "data_format": "'NCL'"
        },
        "paddle_torch_args_map": {
            "x": "input",
            "weight": "weight",
            "bias": "bias",
            "stride": "stride",
            "padding": "padding",
            "dilation": "dilation",
            "groups": "groups"
        }
    },
```

4. 编写转换代码

在 rules.py 中编写 Conv1dRule 类。Rule 只负责 data format、padding 等语义归一化；
`build_result()` 根据 mapping.json 自动生成默认值与参数映射代码。

然后编写 preprocess（预处理）、core（核心执行）、postprocess（后处理）代码块

最终通过 `build_result()` 组装并预编译三个代码阶段，同时显式声明转换类型：

```python
class Conv1dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv1d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
    else:
        padding = tuple(padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        postprocess = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
            postprocess=postprocess,
        )
```

其中 `build_result()` 的 `output_var` 参数默认为 `result`；`kind` 必须显式传入
`ConversionKind.DIRECT` 或 `ConversionKind.COMPOSITE`。

5. 运行测试配置

调用 engineV2.py，paddle.nn.functional.conv1d 的所有测试配置全部通过，至此 Rule 转换完毕！
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_conv1d.txt" --num_gpus=8 --num_workers_per_gpu=1 >> "tester/api_config/test_log/log.log" 2>&1
```

## 结语

感谢同学们仔细阅读 README 至此，如果您有任何修改建议，或问题修复、转换补齐的想法，请提交 Issue 与 PR ，并 at @cangtianhuang 进行 Review

也可以直接发送至开发者邮箱: 1903374751@qq.com / l1903374751@gmail.com

非常感谢以下贡献人员:

@wanghuancoder @cangtianhuang @mzj104 @Cutelemon6 @cszdrg @yuwu46
