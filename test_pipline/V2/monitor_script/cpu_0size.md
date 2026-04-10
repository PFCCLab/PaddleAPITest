# cpu_0size 测试流程

### 1. 准备测试环境

1. 建议在虚拟环境或 docker 中进行开发，并正确安装 python 与 nvidia 驱动，engineV2 建议使用 **python>=3.10**

2. PaddlePaddle 与 PyTorch 的部分依赖项可能发生冲突，请先安装 **paddlepaddle-gpu** 再安装 **torch**，重新安装请添加 `--force-reinstall` 参数

3. 安装 paddlepaddle-gpu

   - [使用 pip 快速安装 paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
   ```
   - 若需要本地编译 Paddle，可参考：[Linux 下使用 ninja 从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile-by-ninja.html)

4. 安装 torch

   - [使用 pip 快速安装 torch](https://pytorch.org/get-started/locally/)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
5. 安装第三方库

   ```bash
   pip install func_timeout pebble nvidia-ml-py
   ```

### 2. 准备测试集
用于监控的0-size测试集: ```tester/api_config/monitor_config/0_size/CPU/monitoring_configs*.txt```
### 3. 准备测试脚本
可使用已有的参考脚本
```test_pipline/V2/monitor_script/run_cpu_0size_full.sh```


### 4. 执行测试
``` bash test_pipline/V2/monitor_script/run_gpu_0size_full.sh ```


最终的所有测试结果会保存在测试脚本中```LOG_DIR```指定的目录下，比如 `tester/api_config/test_log_cpu_0size_full` 目录，包括：
- 检查点文件 checkpoint.txt
- 以 api_config_ 开头的配置集文件
- 测试日志文件 log.log
- 详细测试结果文件 log_inorder.log

### 4. 继续测试

apitest 拥有检查点 checkpoint 机制，保存了所有已经测试过的配置。若希望继续测试，可直接运行脚本，无需重新测试已经测过的配置，**切勿删除测试结果目录**

若存在 skip、oom、crash、timeout 等异常配置，且希望重新测试它们，可使用 `tools/retest_remover.py` 小工具：
- 修改 `TEST_LOG_PATH` 为 `tester/api_config/test_log_cpu_0size_full` （测试脚本中的`LOG_DIR` 指定的输出目录）
- （可选）决定 `LOG_PREFIXES` 中需要重测的配置集
- 直接运行

即可删去 checkpoint.txt 中的配置，然后继续测试

### 5. 整理测试结果

若需要整理出具 error 报告，可使用 `tools/error_stat_0_size.py` 小工具：
- 修改 `TEST_LOG_PATH` 为 `tester/api_config/test_log_cpu_0size_full` （测试脚本中的`LOG_DIR` 指定的输出目录）
- （可选）修改 `# classify logs` 处的日志分类规则，可进行无效日志筛选
- （可选）修改 `# error logs` 处的错误日志列表，可进行错误日志筛选
- 直接运行

即可在原测试结果目录中生成以下文件：
- `error_api.txt`
- `error_config.txt`
- `error_log.log`
- `pass_api.txt`
- `pass_config.txt`
- `pass_log.log`
- `invalid_config.txt`
