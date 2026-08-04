# Agent Guidance

- 默认使用 `engineV4.py` 和仓库现有脚本；只在需要兼容旧流程时考虑 `engineV2.py` 或历史入口。
- 每次修改代码、测试、配置或项目文档后，主动运行项目回归测试：`tools/regression/regression_runner.sh`。
- 回归测试默认使用 `tools/regression/regression_configs.txt`；如需覆盖参数，优先使用脚本支持的环境变量，不要硬编码本机路径或个人目录。
- 单次运行只启用一种主模式，不要把 `--paddle_only`、`--accuracy`、`--accuracy_stable` 等混在一起。
- 新增或修改单条配置行为时，优先用 `engineV4.py --api_config=...` 做最小复现，再回到批量回归验证。
- 如果当前环境缺少 GPU、Paddle 运行时、PyTorch 或必要依赖，最终回复必须明确说明未运行原因，并给出可复现命令。
- 新增或修改 Python、Shell 源码时，暂存区新增非空源码行的注释率必须至少为 10%，提交前运行 `pre-commit run check-added-comment-ratio`。
- 新增注释默认使用中文，重点解释协议边界、失败语义、参数关系和不明显的实现约束；不要用逐行复述代码或无信息量注释凑比例。
