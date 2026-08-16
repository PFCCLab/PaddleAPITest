# Agent Guidance

- 默认从 `engineV4.py` 和仓库现有脚本进入；只有兼容旧流程时才使用 `engineV2.py` 或历史入口。
- 会改变执行结果、设备调度、日志或错误分类的修改，完成后运行 `tools/regression/regression_runner.sh`；仅修改文档、目录或命名时不要求回归。
- 回归默认读取 `tools/regression/regression_configs.txt`；需要调整范围或资源时使用脚本支持的环境变量，禁止写入本机绝对路径。
- 每次引擎运行只选择一个输入源和一个主模式；不要同时启用 `--paddle_only`、`--accuracy`、`--accuracy_stable` 等主模式。
- 修改单条配置行为时，先用 `engineV4.py --api_config=...` 完成最小复现，再运行批量回归。
- GPU、Paddle、PyTorch 或其他依赖缺失时，不要伪造测试结果；最终说明未运行原因，并给出可复现命令。
- 修改 Python 或 Shell 源码时，暂存区新增非空源码行的中文注释率保持至少 10%；提交前运行 `pre-commit run check-added-comment-ratio`。
- 新增注释只解释协议边界、失败语义、参数关系或其他不明显约束；不要逐行复述代码或为凑比例添加空注释。
