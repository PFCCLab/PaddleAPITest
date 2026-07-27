# generic_configs

通用 CI 配置：6 种 `{0size, 1M, 4096} × {accuracy, paddleonly}` 组合，
特殊 accuracy 变体：2 种（`accuracy_compatible`、`accuracy_manual_threshold`），
输入/输出路径用 `${APITEST_MODEL}` 占位，切换模型只需改这一个环境变量。

## 目录约定

需先把模型专属配置拷贝到 `PaddleAPITest/${APITEST_MODEL}/` 下：

```text
${APITEST_MODEL}/
├── accuracy_0size/0size.txt
├── accuracy/1M.txt
├── accuracy/4096.txt
├── paddleonly_0size/0size.txt
├── paddleonly/1M.txt
├── paddleonly/4096.txt
├── accuracy_compatible/accuracy_compatible.txt
└── accuracy_manual_threshold/
    ├── accuracy_manual_threshold.txt
    └── accuracy_manual_threshold_config.yaml
```

## 用法

```bash
export APITEST_MODEL=eb5_1
python run.py -c test_pipeline/generic_configs/run_0size_accuracy.yaml
python run.py -c test_pipeline/generic_configs/run_1M_accuracy.yaml
python run.py -c test_pipeline/generic_configs/run_4096_accuracy.yaml
python run.py -c test_pipeline/generic_configs/run_0size_paddleonly.yaml
python run.py -c test_pipeline/generic_configs/run_1M_paddleonly.yaml
python run.py -c test_pipeline/generic_configs/run_4096_paddleonly.yaml
python run.py -c test_pipeline/generic_configs/run_accuracy_compatible.yaml
python run.py -c test_pipeline/generic_configs/run_accuracy_manual_threshold.yaml
```

也可用 `-i/--input`、`-o/--output` 临时覆盖某次运行的输入文件和日志目录：

```bash
python run.py -c test_pipeline/generic_configs/run_1M_accuracy.yaml \
  -i ${APITEST_MODEL}/accuracy/1M.txt -o test_${APITEST_MODEL}_log_1M_accuracy
```

## num_workers_per_gpu 配置

| 配置 | num_workers_per_gpu |
| --- | --- |
| 0size accuracy | 4 |
| 0size paddleonly | 4 |
| 4096 accuracy | 4 |
| 4096 paddleonly | 4 |
| 1M accuracy/paddleonly | 1 |
| accuracy_compatible / accuracy_manual_threshold | 1 |

## 依赖的 run.py 能力

`run.py` 加载 YAML 后会对所有字符串字段执行 `os.path.expandvars`，
支持 `${VAR}`/`$VAR` 占位符；未设置的变量原样保留。
