# PaddleAPITest Tools

本目录收纳 PaddleAPITest 的配置整理、日志分析、错误统计和专项辅助脚本。以下命令默认在 `PaddleAPITest/` 根目录执行。

## 配置集合工具

- `extract_api_set.py`：从配置 `.txt` 文件或目录中提取 API 名集合，默认输出 `tester/api_config/output/api_extracted.txt`。
  ```bash
  python tools/extract_api_set.py -i tester/api_config/api_config_tmp.txt -o tester/api_config/output
  ```

- `merge_config_set.py`：合并、去重、排序 API 配置集合，可按数量分片输出；原地修改时默认创建 `.backup`。
  ```bash
  python tools/merge_config_set.py -i tester/api_config/configs/ -o tester/api_config/output
  python tools/merge_config_set.py -i config.txt -I
  ```

- `diff_config_set.py`：对比两个配置集合，打印 left 中移除的配置和 right 中新增的配置。
  ```bash
  python tools/diff_config_set.py --left old.txt --right new.txt
  ```

- `retrieve_config_set.py`：按关键词或完整 API 名从配置文件/目录召回配置，默认输出 `tester/api_config/api_config_retrieved.txt`。
  ```bash
  python tools/retrieve_config_set.py -i tester/api_config/5_accuracy -k matmul linear
  python tools/retrieve_config_set.py -i tester/api_config/5_accuracy -k paddle.matmul -e
  ```

- `remove_config_set.py`：按一个或多个配置清单从文件或目录中删除配置；默认仅在文件被修改时，将带时间戳的 `.backup` 和 `.removed_configs` 写入同目录下的 `.rm_config_backups/`。
  ```bash
  python tools/remove_config_set.py -i configs/ -r remove_configs.txt
  python tools/remove_config_set.py -i configs/ -r remove_a.txt remove_b.txt remove_dir/
  python tools/remove_config_set.py -i configs/ -r remove_configs.txt -n
  python tools/remove_config_set.py -i configs/ -m merged_removed.txt
  python tools/remove_config_set.py -i configs/ -R
  python tools/remove_config_set.py -c configs/.rm_config_backups
  ```
  `-R` 回退和 `-c` 精简备份目录会修改已有文件，非 `-n` 模式需要输入 `yes` 二次确认。

- `extract_cases_from_csv.py`：从 CSV 中按 API 名过滤 case，输出 `filtered_result_<api>.csv` 和 `error_config_<api>.txt`。
  ```bash
  python tools/extract_cases_from_csv.py paddle.matmul --config-path TotalStableFull.csv
  python tools/extract_cases_from_csv.py paddle.matmul --only-diff -o output/
  ```

- `extract_or_remove_api_cases.py`：从指定配置目录中按 API 提取 case 到临时文件，可选从源文件删除；删除时默认备份。
  ```bash
  python tools/extract_or_remove_api_cases.py --config paddle.numel --dst mytmp.txt
  python tools/extract_or_remove_api_cases.py --config paddle.numel --remove
  ```

- `remove_lines_by_keyword.py`：按关键词文件删除匹配文件中的行，适合批量清理包含特定关键词的配置；默认备份。
  ```bash
  python tools/remove_lines_by_keyword.py -p 'tester/api_config/monitor_config/accuracy/GPU/monitoring_configs_*.txt' -k kw.txt
  ```

- `random_slim_api_configs.py`：对指定 API 的配置行按稳定随机种子保留指定比例，输出保留集与被裁减集；两份集合按行无交集且并集等于输入。默认处理 `fuse_weighted_swiglu_fp8_quant`、`paddlefleet_fused_swiglu_probs_bwd`、`fp8_quant_blockwise`、`fused_act_dequant`、`moe_permute`、`moe_unpermute`，保留比例为 10%。
  ```bash
  python tools/random_slim_api_configs.py \
    --input-dir configs/original \
    --kept-dir configs/slimmed \
    --removed-dir configs/removed \
    --archive-dir configs/archives \
    --report-file configs/slim_report.json
  ```
  `--api` 可重复指定以覆盖默认 API 列表，`--keep-ratio` 与 `--seed` 可调整保留比例和可复现的随机选择。

- `stratified_slim_preprocessed_configs.py`：针对目录内的每份 `*_preprocessed.txt` 独立按 API 分层裁减，不跨配置集合并抽样。默认保留 API 数量不超过 10 的全部配置；数量更大的 API 保留 `ceil(10% * n)` 条；所有以 `.empty` 或 `.empty_like` 结尾的 API 配置都会被移除。保留集与被裁减补集分别同步生成对应的 `*_api_extracted.txt`。
  ```bash
  python tools/stratified_slim_preprocessed_configs.py \
    --input-dir workspace/0809 \
    --kept-dir workspace/0809_slimmed \
    --removed-dir workspace/0809_slimmed_removed \
    --report-file workspace/0809_slim_report.json
  ```
  `--small-api-threshold`、`--keep-ratio` 与 `--seed` 可调整低频保留阈值、比例和确定性抽样种子。

- `normalize_origin_api_config.py`：历史配置归一化/整理脚本，当前包含较多历史硬编码逻辑；使用前应先确认输入输出路径。

- `shrink_large_configs.py`：针对 paddle_crash、OOM、timeout、config_input 中的大 Tensor 配置缩小 shape，生成缩小后的回归配置。
  ```bash
  python tools/shrink_large_configs.py --error-logs test_log --source-configs source.txt --output shrunk.txt --factor 4
  ```

## 测试日志工具

- `seek_skip_configs.py`：从 `checkpoint.txt` 中扣除已有终态日志，生成 `api_config_skip.txt`，并默认同步更新 checkpoint 与备份。
  ```bash
  python tools/seek_skip_configs.py -p tester/api_config/test_log
  python tools/seek_skip_configs.py -p tester/api_config/test_log --no-update-checkpoint
  ```

- `remove_retest_configs.py`：从 checkpoint 中移除指定结果类型对应配置，并删除对应分类文件；用于重测 timeout、OOM、skip 等集合。
  ```bash
  python tools/remove_retest_configs.py -p tester/api_config/test_log -r timeout oom skip
  ```

- `log_digest.py`：解析原始测试日志，按 pass、accuracy_error、paddle_error、torch_error、cuda_error 等类型拆分日志和配置。
  ```bash
  python tools/log_digest.py --file run.log --id 1 --ckpt_id 1
  ```

- `find_API_config.py`：从日志中提取指定 API 的错误日志和错误配置。
  ```bash
  python tools/find_API_config.py --input-log log.log --api paddle.matmul
  ```

## 错误统计工具

- `error_stat/error_stat.py`：整理 `log_inorder.log` 与 `api_config_*.txt`，输出 pass、skip、Paddle 问题、可重测和框架阻塞分类目录 `error_stat_result/`。
  ```bash
  python tools/error_stat/error_stat.py -i tester/api_config/test_log -o tester/api_config/test_log
  python tools/error_stat/error_stat.py -i tester/api_config/test_log --split-errors
  ```

- `error_stat/csv_stat_tol.py`：整理 `tol*.csv` 精度数据，生成 `tol_full.csv`、`tol_stat.csv`、`tol_stat_api.csv`。
  ```bash
  python tools/error_stat/csv_stat_tol.py -i tester/api_config/test_log
  ```

- `error_stat/csv_stat_stable.py`：整理 `stable*.csv` 稳定性精度数据，生成 `stable_full.csv`、`stable_stat.csv`、`stable_stat_api.csv`。
  ```bash
  python tools/error_stat/csv_stat_stable.py -i tester/api_config/test_log --chunk-size 2000000 --max-workers 10
  ```

- `error_stat/error_summary.py`：错误统计辅助脚本，用于进一步汇总错误信息；使用前建议查看脚本内参数说明。

## API Trace 与专项工具

- `api_tracer/api_tracer.py`：通过 monkey-patch 捕获框架 API 调用并序列化为 PaddleAPITest 配置；详见 `tools/api_tracer/README.md`。

- `api_tracer/api_alias_tool.py`、`api_tracer/api_map_tool.py`、`api_tracer/api_merge_tool.py`：API trace 配套的 alias、mapping、merge 辅助工具。

- `prof/`：单 API 性能 demo 与 profiling 脚本目录，输出结果通常是本地分析产物，不建议提交。

- `accuracy_demo.py`、`performance_demo.py`：手工调试 accuracy / performance 路径的示例脚本。

- `prof_api_gsb.py`：性能 API 分组/标记辅助脚本，当前主要作为静态数据脚本使用。

- `test_signature.py`、`test_tool.py`：开发期辅助测试脚本，使用前先确认当前内容是否符合目标场景。

## 注意事项

- 会原地修改配置或 checkpoint 的工具默认创建 `.backup`；如确认不需要备份，可使用对应脚本的 `--no-backup`。
