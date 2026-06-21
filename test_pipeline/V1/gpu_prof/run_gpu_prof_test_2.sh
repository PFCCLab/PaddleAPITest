export CUDA_VISIBLE_DEVICES=1
# sleep 3h
for i in {1..10000}; do python engine.py --api_config_file=test_pipeline/V1/gpu_prof/gpu_prof_test_config_2.txt --paddle_torch_gpu_performance=True >> test_pipeline/V1/gpu_prof/gpu_prof_test_config_2.log 2>&1; done
